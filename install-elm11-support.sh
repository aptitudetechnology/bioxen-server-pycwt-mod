#!/usr/bin/env bash
# Install pyserial for ELM11 support

echo "Installing pyserial for ELM11 FPGA communication..."
pip install pyserial

echo ""
echo "Testing ELM11 detection..."
python test-elm11.py
