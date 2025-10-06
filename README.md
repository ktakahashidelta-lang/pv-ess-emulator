# PV+ESS Modbus Emulator (D1J-style) — Streamlit

This repo contains:
- `d1j_emulator.py` — D1J-style Modbus emulator (Base-1 addressing). Can run as a real Modbus TCP server **locally/VPS**.
- `app.py` — Streamlit dashboard that **simulates** the plant and shows key registers (no external TCP exposure on Streamlit Cloud).
- `requirements.txt` — dependencies.

## Local Run (TCP server)
```bash
pip install -r requirements.txt
python d1j_emulator.py --host 0.0.0.0 --port 1502 --unit 5 --pv-kwp 10
```

## Streamlit (dashboard)
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. On https://share.streamlit.io/ choose "New app", select the repo/branch, and set `app.py` as the entrypoint.
3. Done. (Remember: Streamlit Cloud does not expose raw TCP port 1502 to the Internet.)

## Exposing Modbus TCP to external clients
Use a VPS (e.g., Lightsail/EC2) or any host where you can open port **1502**:
```bash
sudo ufw allow 1502/tcp
python d1j_emulator.py --host 0.0.0.0 --port 1502 --unit 5
```
For production, create a systemd service.
