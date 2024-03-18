from flask import Flask, render_template, Response, send_from_directory
import cv2 as cv
import numpy as np
import os
import datetime
import time
import discord
from discord.ext import commands, tasks
import threading
import asyncio

application_id = '1212064029921779722'
public_key = '00e95274adf8b36f31d559c6ef9dd817a59f2d50c5e970124424ce51aed1fa90'
bot_token = 'MTIxMjA2NDAyOTkyMTc3OTcyMg.GkSLNE.1K3ZuXSCVcook1izfA4i4kn8GRORj2yxM4fOOE'
CHANNEL_ID = 1103720652147535956

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

video_output_path = 'recorded_videos'
os.makedirs(video_output_path,exist_ok=True)
face_cascade = cv.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
body_cascade = cv.CascadeClassifier('./cascades/haarcascade_upperbody.xml')
eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_eye.xml')
hand_cascade = cv.CascadeClassifier('./cascades/hand.xml')


async def generate_frames(channel):
    video_feed = cv.VideoCapture(0)
    frame_size = (int(video_feed.get(3)), int(video_feed.get(4)))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 5
    
    while True:
        success, frame = video_feed.read()
        if not success:
            break
        
        grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        bodies = body_cascade.detectMultiScale(grayscale_frame,scaleFactor=1.3,minNeighbors=3,minSize = (30,30))
        eyes = eye_cascade.detectMultiScale(grayscale_frame,scaleFactor=1.3,minNeighbors=3,minSize = (30,30))
        hands = hand_cascade.detectMultiScale(grayscale_frame,scaleFactor=1.3,minNeighbors=3,minSize = (30,30))
        
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
            
        for x,y,w,h in eyes:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            
        for x,y,w,h in hands:
            cv.rectangle(frame,(x+y),(x+w,y+h),(0,0,255),4)
        if len(faces)+len(hands)+len(eyes) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out = cv.VideoWriter(
                    f"{video_output_path}/{current_time}.mp4", fourcc, 20.0, frame_size)
                await channel.send(f"Face detected at {current_time.split("_")[0]} at {current_time.split("_")[1].replace("-",":")}")
                print("Started recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    await channel.send(file=discord.File(f"{video_output_path}/{current_time}.mp4"))
                    await channel.send("Stopped recording!")
                    
                    print("Stopped recording!")
            else:
                timer_started = True
                detection_stopped_time = time.time()
        
        if detection:
            out.write(frame)  


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')


@bot.command()
async def start_surveillance(ctx):
    # Start generating frames and detections
    channel = ctx.channel
    
    await channel.send('Starting surveillance...')
    await generate_frames(channel)

bot.run(bot_token)

