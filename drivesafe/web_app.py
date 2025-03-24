import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import torch.nn as nn
from flask import Flask, render_template, Response, request, redirect, url_for
import threading
import queue

# Create Flask app with correct template and static folders
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
