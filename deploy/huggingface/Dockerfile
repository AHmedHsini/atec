FROM python:3.13-slim

RUN useradd -m -u 1000 user
RUN pip install --no-cache-dir numpy==2.4.4 opencv-python-headless==4.13.0.92

WORKDIR /app
COPY --chown=user veil/run_mission.py veil/requirements.txt /app/veil/
COPY --chown=user veil/veil /app/veil/veil
COPY --chown=user veil/web /app/veil/web
COPY --chown=user veil/outputs/hazard_net.npz veil/outputs/hazard_net_polar.npz /app/veil/outputs/
COPY --chown=user dataset/panorama /app/dataset/panorama

USER user
WORKDIR /app/veil
ENV PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "run_mission.py", "--panorama", "--in-process", "--loss", "0", "--host", "0.0.0.0"]
