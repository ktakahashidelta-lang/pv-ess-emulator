
import time
import streamlit as st
try:
    from d1j_emulator import D1JServer, PlantModel
except Exception as e:
    st.title("PV+ESS Modbus Emulator (D1J-style)")
    st.error("⚠️ 'pymodbus' または依存モジュールの読み込みに失敗しました。")
    st.info("requirements.txt を以下の内容に修正し、再デプロイしてください。\n\n"
            "```\n"
            "streamlit==1.36.0\n"
            "pymodbus==3.6.8\n"
            "pandas==2.2.2\n"
            "```")
    st.exception(e)
    st.stop()
           
st.set_page_config(page_title="PV+ESS Modbus Emulator (D1J-style)", layout="wide")

st.title("PV+ESS Modbus Emulator (D1J-style) — Streamlit Dashboard")
st.caption("Note: Streamlit Cloud does not expose raw TCP ports. This app runs the model and shows live registers.\n"
           "For an actual Modbus TCP server reachable from outside, deploy the same code on a VPS and open port 1502.")

if "server" not in st.session_state:
    model = PlantModel(pv_kwp=6.0)
    st.session_state.server = D1JServer(unit=5, model=model)
    st.session_state.running = True
    st.session_state.last_tick = time.time()
    st.session_state.timestep = 1.0

server: D1JServer = st.session_state.server

st.sidebar.header("Simulation Controls")
pv_kwp = st.sidebar.number_input("PV Nameplate (kWp)", min_value=0.0, max_value=100.0, value=server.model.pv_kwp, step=0.5)
server.model.pv_kwp = pv_kwp

weather = st.sidebar.slider("Weather Scalar", 0.0, 2.0, server.model.weather, 0.1)
server.model.weather = weather

pm_all = st.sidebar.selectbox("PM Mode (All)", options=["Disable (follow indiv.)", "Rated (use % below)"], index=server.model.pm_mode_all)
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

now = time.time()
if st.session_state.running and (now - st.session_state.last_tick) >= st.session_state.timestep:
    server.tick()
    st.session_state.last_tick = now

m = server.model
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AC Total Power (kW)", f"{m.ac_total_power_kw:.2f}")
c2.metric("DC Total Power (kW)", f"{m.dc_total_power_kw:.2f}")
c3.metric("PV Energy Today (kWh)", f"{m.today_energy_kwh:.2f}")
c4.metric("Life Energy (MWh)", f"{m.life_energy_kwh/1000.0:.2f}")
state_txt = "RUN" if m.state == 2 else "STANDBY"
c5.metric("State", state_txt)

st.subheader("Key Registers (Base-1 addressing)")
def rd(addr, n=1):
    return server.ir.getValues(addr, n)

rows = []
def add_row(addr, label, note=""):
    val = rd(addr, 1)[0]
    rows.append((addr, label, val, note))

add_row(10148, "AC Total Power (0.1kW)", "divide by 10")
add_row(10161, "DC Total Power (0.1kW)", "divide by 10")
add_row(10149, "AC Voltage L1 (0.1V)", "divide by 10")
add_row(10150, "AC Current L1 (0.1A)", "divide by 10")
add_row(10152, "AC Frequency L1 (0.01Hz)", "divide by 100")
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
df = pd.DataFrame(rows, columns=["Address", "Register", "Raw", "Note"])
st.dataframe(df, use_container_width=True)

st.caption("Auto-refresh is tied to the 'Run simulation' toggle and 'Timestep (sec)'.")
