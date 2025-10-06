#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta D1J-style PV Inverter Modbus TCP Emulator
===============================================
Imitates major parts of the "D1J Protocol Definition" (Base-1 Modbus addresses).
- Modbus TCP
- PLC Addresses Mode (Base 1)
- Key addresses around 700~, 101xx~, 1026x~, 104xx~, 105xx~, 106xx~, 1070x~

Install:
  pip install "pymodbus==3.*"

Run:
  python d1j_emulator.py --host 0.0.0.0 --port 1502 --unit 5 --pv-kwp 10 --log INFO

Notes:
- Base-1 addressing per spec. Read exactly e.g. address 10148 for AC Total Power, etc.
- Scale factors (10109..10113) are provided; main values are scaled accordingly.
- This is a lightweight functional emulator; values are simulated and NOT vendor-approved.
"""
import argparse
import logging
import math
import random
import signal
import sys
import threading
import time
from typing import Dict, List

from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.transaction import ModbusSocketFramer

log = logging.getLogger("d1j_emulator")

# ---------------------------- Utilities ----------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def u16(x):  # unsigned 16-bit
    return x & 0xFFFF

def s16_to_u(val):  # signed int -> two's complement
    return val & 0xFFFF

def u_to_s16(val):  # two's complement -> signed int
    return val - 0x10000 if val & 0x8000 else val

# ---------------------------- Plant model ----------------------------

class PlantModel:
    """Simple PV + Load model; energies and faults are synthesized."""
    def __init__(self, pv_kwp=6.0, init_time_min=9*60):
        self.minute = init_time_min  # 0..1439
        self.weather = 1.0
        self.pv_kwp = pv_kwp
        self.load_kw = 1.2
        self.ac_total_power_kw = 0.0
        self.dc_total_power_kw = 0.0
        self.today_energy_kwh = 0.0
        self.life_energy_kwh = 10000.0  # seed
        self.ac_v_phase = [230.0, 230.0, 230.0]
        self.ac_i_phase = [2.0, 2.0, 2.0]
        self.ac_f_phase = [50.0, 50.0, 50.0]
        self.temps = [35]*10
        self.state = 2  # STATUS_ON
        self.fault_bits = [0]*8  # 10260..10267
        self.error_bits = [0]*5  # 10268..10272
        self.warn_bits  = [0]*2  # 10273..10274

        # PM (Power Management) writable
        self.pm_mode_all = 0   # 700 (0: Disable, 1: Rated)
        self.pm_percent_all = 100  # 701 (0..100)
        self.grid_lock = 0     # 706 (1: unlock pulse)

        self.pm_mode = 0       # 10700
        self.pm_percent = 100  # 10701

        # Scale factors
        self.volt_sf = 1   # 10109 (multiplier, interpret as 0.1V when 0.1 factor used elsewhere)
        self.idc_sf = 1    # 10110
        self.iac_sf = 1    # 10111
        self.pwr_sf = 1    # 10112
        self.eng_sf = 1    # 10113

    def _pv_profile_kw(self):
        t = self.minute/60.0
        if t < 6 or t > 18.5:
            return 0.0
        x = (t-6.0)/(18.5-6.0)*math.pi
        g = math.sin(x)**1.4
        return max(0.0, g) * self.pv_kwp * self.weather

    def _load_profile_kw(self):
        # random walk with occasional spikes
        self.load_kw = clamp(self.load_kw + random.uniform(-0.05, 0.05), 0.5, 3.5)
        if random.random() < 0.01:
            self.load_kw = min(self.load_kw + random.uniform(0.5, 1.5), 5.0)
        return self.load_kw

    def step(self, dt_s=1.0):
        pv_kw = self._pv_profile_kw()
        load_kw = self._load_profile_kw()

        # Apply PM percent (caps active power output)
        pm_percent = self.pm_percent_all if self.pm_mode_all == 1 else self.pm_percent
        pm_percent = clamp(pm_percent, 0, 100)
        ac_out_kw = min(pv_kw, pv_kw*pm_percent/100.0)

        self.ac_total_power_kw = ac_out_kw
        self.dc_total_power_kw = max(0.0, pv_kw)  # simplistic

        # simple three‑phase currents assuming 230V L-N
        for i in range(3):
            self.ac_v_phase[i] = 230.0 + random.uniform(-2.0, 2.0)
            self.ac_f_phase[i] = 50.0 + random.uniform(-0.05, 0.05)
            self.ac_i_phase[i] = (ac_out_kw/3.0) / (self.ac_v_phase[i]/1000.0)

        # temperatures drift
        for i in range(10):
            self.temps[i] = clamp(self.temps[i] + random.uniform(-0.2, 0.2), 20.0, 75.0)

        # energies (kWh)
        self.today_energy_kwh += self.ac_total_power_kw * (dt_s/3600.0)
        self.life_energy_kwh  += self.ac_total_power_kw * (dt_s/3600.0)

        # state: ON if pv>0.1kW else STANDBY
        self.state = 2 if pv_kw > 0.1 else 0

        # time
        self.minute = (self.minute + int(dt_s/60.0)) % (24*60)

# ---------------------------- Register server ----------------------------

class D1JDataBlocks(ModbusSequentialDataBlock):
    """Large address space, Base-1 addressing (zero_mode=False)."""
    def __init__(self, name: str, size: int = 12000):
        super().__init__(address=1, values=[0]*size)  # start at 1 to align with Base-1
        self.name = name

class D1JServer:
    def __init__(self, unit: int, model: PlantModel):
        self.unit = unit
        self.model = model
        # massive maps to cover up to ~11k addresses
        self.hr = D1JDataBlocks("HR", size=12050)
        self.ir = D1JDataBlocks("IR", size=12050)
        self.co = D1JDataBlocks("CO", size=100)   # not used, provided for completeness
        self.di = D1JDataBlocks("DI", size=100)

        # Initialize constants / scale factors
        # 10109..10113 scale factors -> here we just set 1 (engineering units already scaled)
        for addr in (10109,10110,10111,10112,10113):
            self.ir.setValues(addr, [1])

        # Example model name 11000..11007 (ASCII pairs)
        model_ascii = "Delta-PCS".ljust(16)[:16].encode("ascii")
        for i in range(0, 16, 2):
            word = (model_ascii[i] << 8) | model_ascii[i+1]
            self.ir.setValues(11000 + i//2, [word])

        # Seed life energy words 10196..10198 (Wh, 3 words: low, mid, high)
        self._write_life_energy_wh(int(self.model.life_energy_kwh * 1000))

        # PM initial
        self.hr.setValues(700, [0])     # PM Mode Set All: 0 Disable
        self.hr.setValues(701, [100])   # PM Percent Set All
        self.hr.setValues(706, [0])     # Unlock Grid Lock
        self.hr.setValues(10700, [0])   # PM Mode
        self.hr.setValues(10701, [100]) # PM Percent

    # ---- Helpers to split/join 48-bit energies across 3 words (wh) ----
    def _write_energy_2w(self, base_addr: int, wh: int):
        # Some tables use 2-word (low, high). We'll fill both anyway.
        lo = wh & 0xFFFF
        hi = (wh >> 16) & 0xFFFF
        self.ir.setValues(base_addr,   [lo])
        self.ir.setValues(base_addr+1, [hi])

    def _write_life_energy_wh(self, wh: int):
        # 3 words: low, middle, high (per table 10196..10198)
        w0 =  wh        & 0xFFFF
        w1 = (wh >> 16) & 0xFFFF
        w2 = (wh >> 32) & 0xFFFF
        self.ir.setValues(10196, [w0])
        self.ir.setValues(10197, [w1])
        self.ir.setValues(10198, [w2])

    # ---- Periodic publish from model to registers ----
    def publish(self):
        m = self.model

        # 10148 AC Total Power (scaled by Power SF; we keep kW -> assume 0.1kW units here)
        self.ir.setValues(10148, [u16(int(round(m.ac_total_power_kw*10)))])

        # 10149..10160 AC Volt/Current/Watt/Freq per phase (simplified)
        # We'll publish Phase‑1 entries only (others exist but may remain 0 or same)
        self.ir.setValues(10149, [u16(int(round(m.ac_v_phase[0]*10)))])   # 0.1V
        self.ir.setValues(10150, [u16(int(round(m.ac_i_phase[0]*10)))])   # 0.1A (assumed)
        self.ir.setValues(10151, [u16(int(round((m.ac_total_power_kw/3.0)*10)))])  # 0.1kW per phase
        self.ir.setValues(10152, [u16(int(round(m.ac_f_phase[0]*100)))])  # 0.01Hz

        # DC Total Power 10161 (0.1kW)
        self.ir.setValues(10161, [u16(int(round(m.dc_total_power_kw*10)))])

        # Temperatures 10180..10190 (1℃)
        for i in range(10):
            self.ir.setValues(10180+i, [u16(int(round(m.temps[i])))])
        self.ir.setValues(10190, [u16(int(round(max(m.temps))))])  # Max

        # Today energy 10192..10193 (Wh, low/high)
        today_wh = int(round(m.today_energy_kwh*1000))
        self._write_energy_2w(10192, today_wh)

        # Inverter State 10194
        # STATUS_ON = 2, STANDBY = 0
        self.ir.setValues(10194, [2 if m.state == 2 else 0])

        # Life energy 10196..10198
        life_wh = int(round(m.life_energy_kwh*1000))
        self._write_life_energy_wh(life_wh)

        # Fault/Error/Warning bitfields 10260..10275
        # Keep zeros; inject a transient warning occasionally
        if random.random() < 0.005:
            self.model.warn_bits[0] ^= 0x0002  # toggle W02 bit1
        for i, val in enumerate(self.model.fault_bits):
            self.ir.setValues(10260+i, [u16(val)])
        for i, val in enumerate(self.model.error_bits):
            self.ir.setValues(10268+i, [u16(val)])
        for i, val in enumerate(self.model.warn_bits):
            self.ir.setValues(10273+i, [u16(val)])

        # Daily energy ring 10400..10463 (Day-0..Day-31, each 2 words Wh)
        # Here we synthesize Day-0 only from today_wh; others 0
        self._write_energy_2w(10400, today_wh)
        for off in range(2, 64, 2):
            self._write_energy_2w(10400+off, 0)

        # Monthly 10500..10547 and Yearly 10600..10639 (2 words each)
        self._write_energy_2w(10500, life_wh % (1000*1000))  # dummy smaller number
        for off in range(2, 48, 2):
            self._write_energy_2w(10500+off, 0)
        self._write_energy_2w(10600, life_wh % (10_000_000))
        for off in range(2, 40, 2):
            self._write_energy_2w(10600+off, 0)

        # PM writable mirrors
        self.ir.setValues(700, [u16(self.model.pm_mode_all)])
        self.ir.setValues(701, [u16(self.model.pm_percent_all)])
        self.ir.setValues(10700, [u16(self.model.pm_mode)])
        self.ir.setValues(10701, [u16(self.model.pm_percent)])

        # Status bitmap example 210 (Inverter1 Status) – set "On" (Bit9) when state ON
        # Bit9=On -> 1<<9 = 0x0200
        inv1_status = 0x0200 if m.state == 2 else 0
        self.ir.setValues(210, [u16(inv1_status)])

    # ---- Process writes to HR region for PM / unlock ----
    def apply_writes(self):
        # Read back command words and update model
        pm_mode_all = self.hr.getValues(700, 1)[0] & 0xFFFF
        pm_percent_all = self.hr.getValues(701, 1)[0] & 0xFFFF
        grid_unlock = self.hr.getValues(706, 1)[0] & 0xFFFF

        pm_mode = self.hr.getValues(10700, 1)[0] & 0xFFFF
        pm_percent = self.hr.getValues(10701, 1)[0] & 0xFFFF

        self.model.pm_mode_all = 1 if pm_mode_all == 1 else 0
        self.model.pm_percent_all = clamp(pm_percent_all, 0, 100)
        if grid_unlock == 1:
            # emulate a pulse: clear immediately
            self.hr.setValues(706, [0])

        self.model.pm_mode = 1 if pm_mode == 1 else 0
        self.model.pm_percent = clamp(pm_percent, 0, 100)

    def tick(self):
        self.apply_writes()
        self.model.step(1.0)
        self.publish()

# ---------------------------- Server runner ----------------------------

def start_loop(servers: Dict[int, D1JServer], timestep: float):
    stop = [False]
    def loop():
        while not stop[0]:
            for srv in servers.values():
                srv.tick()
            time.sleep(max(0.2, timestep))
    t = threading.Thread(target=loop, name="d1j-sim", daemon=True)
    t.start()
    return stop

def main():
    ap = argparse.ArgumentParser(description="Delta D1J-style Modbus TCP Emulator (Base-1)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1502)
    ap.add_argument("--unit", type=int, action="append", default=[5], help="Unit ID(s) (D1J inverter ID)")
    ap.add_argument("--pv-kwp", type=float, default=6.0)
    ap.add_argument("--timestep", type=float, default=1.0)
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    servers: Dict[int, D1JServer] = {}
    slaves: Dict[int, ModbusSlaveContext] = {}
    for uid in args.unit:
        model = PlantModel(pv_kwp=args.pv_kwp)
        srv = D1JServer(uid, model)
        servers[uid] = srv
        slaves[uid] = ModbusSlaveContext(
            di=srv.di, co=srv.co, hr=srv.hr, ir=srv.ir, zero_mode=False  # Base‑1 addressing
        )

    context = ModbusServerContext(slaves=slaves, single=False)

    ident = ModbusDeviceIdentification()
    ident.VendorName = "DEMO-D1J"
    ident.ProductCode = "D1J-EMU"
    ident.VendorUrl = "https://example.invalid"
    ident.ProductName = "D1J Protocol Emulator"
    ident.ModelName = "D1J-Emu"
    ident.MajorMinorRevision = "1.0"

    stop = start_loop(servers, args.timestep)

    def on_sig(signum, frame):
        log.info("Signal %s -> exit", signum)
        stop[0] = True
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    StartTcpServer(context=context, identity=ident, address=(args.host, args.port), framer=ModbusSocketFramer)

if __name__ == "__main__":
    main()
