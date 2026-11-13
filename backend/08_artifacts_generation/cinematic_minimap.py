"""Animated Standard floor plan synchronized to sampled tour frames."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_interactive_viewer import project


class AnimatedMinimap:
    def __init__(self, viewer_dir, tour, flight, floors, width, hfov_deg=70):
        self.flight = flight
        self.floors = np.asarray(floors)
        self.hfov = math.radians(hfov_deg)
        self.size = max(80, round(width * .42))
        self.index = -1
        self.checkpoints = {c['number']: c for c in tour['cameras']}
        self.events = {event['frame']: self.checkpoints[event['number']]
                       for event in flight['checkpoints']}
        self.plans, self.grids, self.progress, self.markers = {}, {}, {}, {}
        self.font = ImageFont.truetype('DejaVuSans-Bold.ttf', max(9, round(self.size * .055)))
        for level in tour['levels']:
            name = level['name']
            info = level['styles']['objects']
            path = Path(viewer_dir) / info['image'].split('?', 1)[0]
            self.grids[name] = info['grid']
            background = Image.open(path).convert('RGB').resize((self.size, self.size), Image.Resampling.LANCZOS)
            layer = Image.new('RGBA', (self.size, self.size))
            draw = ImageDraw.Draw(layer)
            coords = [self.pixel(p, name) for p in flight['camera_to_world'][:, :3, 3]]
            start = None
            for i, floor in enumerate(self.floors):
                if floor == name and start is None:
                    start = i
                if start is not None and (floor != name or i == len(self.floors) - 1):
                    end = i if floor != name else i + 1
                    if end - start > 1:
                        draw.line(coords[start:end], fill=(20, 100, 230, 85), width=max(2, self.size // 90), joint='curve')
                    start = None
            background = Image.alpha_composite(background.convert('RGBA'), layer)
            self.plans[name] = background
            self.progress[name] = Image.new('RGBA', (self.size, self.size))
            self.markers[name] = Image.new('RGBA', (self.size, self.size))

    def pixel(self, position, floor):
        grid = self.grids[floor]
        x, y = project(position, grid)
        return (x * self.size / grid['width'], y * self.size / grid['height'])

    def advance(self, index):
        if index < self.index or index >= len(self.floors):
            raise ValueError('Minimap frames must advance in order')
        for i in range(self.index + 1, index + 1):
            floor = self.floors[i]
            position = self.flight['camera_to_world'][i, :3, 3]
            if i and self.floors[i-1] == floor:
                previous = self.flight['camera_to_world'][i-1, :3, 3]
                draw = ImageDraw.Draw(self.progress[floor])
                draw.line([self.pixel(previous, floor), self.pixel(position, floor)],
                          fill=(10, 90, 210, 255), width=max(2, self.size // 80))
            camera = self.events.get(i)
            if camera:
                x, y = self.pixel(camera['position'], floor)
                radius = max(6, self.size * .035)
                draw = ImageDraw.Draw(self.markers[floor])
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(10, 80, 180, 255),
                             outline='white', width=max(1, self.size // 150))
                draw.text((x, y), str(camera['number']), font=self.font, anchor='mm', fill='white')
        self.index = index
        floor = self.floors[index]
        result = Image.alpha_composite(self.plans[floor], self.progress[floor])
        result = Image.alpha_composite(result, self.markers[floor])
        overlay = Image.new('RGBA', result.size)
        draw = ImageDraw.Draw(overlay)
        pose = self.flight['camera_to_world'][index]
        x, y = self.pixel(pose[:3, 3], floor)
        forward = pose[:3, 2]
        if np.linalg.norm(forward[[0, 2]]) < .1:
            return result.convert('RGB')
        yaw = math.atan2(float(forward[2]), float(forward[0]))
        radius = self.size * .20
        cone = [(x, y)] + [(x + radius * math.cos(yaw + angle), y + radius * math.sin(yaw + angle))
                            for angle in (-self.hfov / 2, self.hfov / 2)]
        draw.polygon(cone, fill=(30, 115, 225, 85), outline=(20, 80, 190, 210))
        r = max(4, self.size * .028)
        tip = (x + 1.6*r*math.cos(yaw), y + 1.6*r*math.sin(yaw))
        left = (x + r*math.cos(yaw + 2.5), y + r*math.sin(yaw + 2.5))
        right = (x + r*math.cos(yaw - 2.5), y + r*math.sin(yaw - 2.5))
        draw.polygon((tip, left, right), fill=(8, 45, 105, 255), outline='white', width=1)
        return Image.alpha_composite(result, overlay).convert('RGB')


def compose_minimap(rgb, minimap):
    minimap_arr = np.asarray(minimap)
    mh, mw = minimap_arr.shape[:2]
    pad = max(3, round(rgb.shape[1] * .015))
    out = rgb if rgb.flags.writeable else rgb.copy()
    out[pad:pad + mh, -pad - mw:-pad] = minimap_arr[:, :, :3]
    return out

