FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv g++ mafft ncbi-blast+ progressivemauve && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/nhejscan
COPY . .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir . pytest
ENV PATH=/opt/venv/bin:$PATH
RUN python3 -c "from nhejscan.build import ensure_trace_ext; ensure_trace_ext()"
ENTRYPOINT ["nhejscan"]
