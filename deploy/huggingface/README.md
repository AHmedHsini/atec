---
title: VEIL Mission Control
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: ATEC'26 lunar scout, live on an Apollo 11 panorama
---

# VEIL Mission Control

Live demo of ATEC'26 Project C.R.A.T.E.R. A software lunar scout looks around the Apollo 11 landing site, detects rocks and pits onboard, and sends each one to the lunar hub as a sealed 20-byte packet.

Drag the photo or press the arrow keys to turn the camera. Each new heading is a new frame run through the PolarDB-trained network (0.581 held-out accuracy on PolarDB terrain).

One mission is shared by every visitor, so turning the camera turns it for everyone watching.

Source code, architecture, and training details: https://github.com/AHmedHsini/atec
