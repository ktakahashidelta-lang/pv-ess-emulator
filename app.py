import time
import streamlit as st

# ✅ 最初の Streamlit コールは set_page_config にする
st.set_page_config(page_title="PV+ESS Modbus Emulator (D1J-style)", layout="wide")

# --- pymodbus / d1j_emulator の読み込みを安全に ---
try:
    from d1j_emulator import D1JServer, PlantModel
    pymodbus_status = "✅ pymodbus import OK"
except Exception as e:
    st.title("PV+ESS Modbus Emulator (D1J-style)")
    st.error("⚠️ 'pymodbus' または依存モジュールの読み込みに失敗しました。")
    st.info(
        "requirements.txt は次を推奨:\n\n"
        "```\n"
        "streamlit==1.36.0\n"
        "pymodbus==3.6.8\n"
        "pandas==2.2.2\n"
        "```"
    )
    st.write("エラー詳細:")
    st.exception(e)
    st.stop()

st.title("PV+ESS Modbus Emulator (D1J-style) — Streamlit Dashboard")
st.caption(
    "Note: Streamlit Cloud does not expose raw TCP ports. "
    "This app runs the model and shows live registers. "
    "For an actual Modbus TCP server reachable from outside, deploy on a VPS and open port 1502."
)
st.caption(pymodbus_status)

# ---- セッション初期化 ----
if "server" not in st.session_state:
    model = PlantModel(pv_kwp=6.0)
    st.session_state.server = D1JServer(unit=5, model=model)
    st.session_state.running = True
    st.session_state.last_tick = time.time()
    st.session_state.timestep = 1.0

server: D1JServer = st.session_state.server

# ---- サイドバー操作 ----
st.sidebar.header("Simulation Controls")
pv_kwp = st.sidebar.number_input("PV Nameplate (kWp)", 0.0, 100.0, server.model.pv_kwp, 0.5)
server.model.pv_kwp = pv_kwp

weather = st.sidebar.slider("Weather Scalar", 0.0, 2.0, server.model.weather, 0.1)
server.model.weather = weather

pm_all = st.sidebar.selectbox(
    "PM Mode (All)",
    options=["Disable (follow indiv.)", "Rated (use % below)"],
    index=server.model.pm_mode_all,
)
server.model.pm_mode_all = 1 if pm_all.startswith("Rated") else 0

pm_percent_all = st.sidebar.slider("PM Percent (All)", 0, 100, server.model.pm_percent_all, 1)
server.model.pm_percent_all = pm_percent_all

pm_indiv_on = st.sidebar.checkbox("Individual PM ON (10700)", value=server.model.pm_mode == 1)
server.model.pm_mode = 1 if pm_indiv_on else 0

pm_indiv = st.sidebar.slider("PM Percent (10701)", 0, 100, server.model.pm_percent, 1)
server.model.pm_percent = pm_indiv

timestep = st.sidebar.slider("Timestep (sec)", 0.2, 2.0, float(st.session_state.timestep), 0.1)
st.session_state.timestep = timestep

run_toggle = st.sidebar.toggle("Run simulation", value=st.session_state.running)
st.session_state.running = run_toggle

# ---- 1ステップ進める ----
now = time.time()
if st.session_state.running and (now - st.session_state.last_tick) >= st.session_state.timestep:
    server.tick()
    st.session_state.last_tick = now

# ---- メトリクス表示 ----
m = server.model
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AC Total Power (kW)", f"{m.ac_total_power_kw:.2f}")
c2.metric("DC Total Power (kW)", f"{m.dc_total_power_kw:.2f}")
c3.metric("PV Energy Today (kWh)", f"{m.today_energy_kwh:.2f}")
c4.metric("Life Energy (MWh)", f"{m.life_energy_kwh/1000.0:.2f}")
c5.metric("State", "RUN" if m.state == 2 else "STANDBY")

# ---- キーレジスタ ----
st.subheader("Key Registers (Base-1 addressing)")
def rd(addr, n=1): return server.ir.getValues(addr, n)

rows = []
def add_row(addr, label, note=""):
    rows.append((addr, label, rd(addr, 1)[0], note))

add_row(10148, "AC Total Power (0.1kW)", "÷10")
add_row(10161, "DC Total Power (0.1kW)", "÷10")
add_row(10149, "AC Voltage L1 (0.1V)", "÷10")
add_row(10150, "AC Current L1 (0.1A)", "÷10")
add_row(10152, "AC Frequency L1 (0.01Hz)", "÷100")
add_row(10192, "Today Energy Low (Wh)")
add_row(10193, "Today Energy High (Wh)")
add_row(10194, "State (0=Standby,2=On)")
add_row(10196, "Life Energy W0 (Wh)")
add_row(10197, "Life Energy W1 (Wh)")
add_row(10198, "Life Energy W2 (Wh)")
add_row(700, "PM Mode All (0/1)")
add_row(701, "PM Percent All (%)")
add_row(10700, "PM Mode (0/1)")
add_row(10701, "PM Percent (%)")

import pandas as pd
st.dataframe(pd.DataFrame(rows, columns=["Address", "Register", "Raw", "Note"]), use_container_width=True)

st.caption("If you still see errors, open Manage app → Logs and share the first 10 lines here.")
