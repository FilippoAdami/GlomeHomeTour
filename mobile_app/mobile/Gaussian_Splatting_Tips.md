# Gaussian Splatting Guide: Data Acquisition, Processing & Editing

**Phase 1: Data Acquisition (Blackmagic Camera App)**

* **Codec (H.265 vs. H.264):**
* *Tip:* Use **H.265** encoding.


* *Logic:* H.265 supports 4K at 60 frames per second while maintaining smaller file sizes. The **60 FPS** readout reduces rolling shutter distortion, helping the captured frames better match the mathematical projection models used by reconstruction software.


* *Note:* Use **H.264** only if you plan to feed footage into specialized third-party extraction tools like *Sharp Frame*, as some web utilities require H.264 compatibility.




* **Bitrate & Resolution:**
* *Tip:* Set resolution to **4K** and the bitrate to the **maximum value**.


* *Logic:* Even if fine compression artifacts are invisible to the naked human eye, higher bitrates preserve subtle pixel data critical for stereo matching and feature tracking.




* **Time-Lapse Mode:**
* *Tip:* Enable time-lapse capture (e.g., saving 1 frame every 10 frames).


* *Logic:* Continuous full-rate video is computationally redundant and fills up mobile storage. Interval capture drastically reduces file size (~10x smaller) and mitigates smartphone overheating during long scanning sessions.




* **Lens Distortion Correction:**
* *Tip:* **Turn off** built-in lens correction.


* *Logic:* Photogrammetry and Gaussian Splatting algorithms run their own mathematical calibration of optical distortion. In-camera computational corrections interfere with the true physical optical path, breaking solver assumptions.




* **Digital Stabilization:**
* *Tip:* **Turn off** software/digital stabilization (use a handheld gimbal for physical stabilization instead).


* *Logic:* Digital stabilization constantly dynamically crops, pans, zooms, and rotates the image frame, which continuously shifts the camera’s intrinsic parameters (focal length and principal point). Maintaining fixed intrinsics avoids overfitting during pose calculation.




* **Lens Choice & Focus Strategy:**
* *Ultra-wide Lens:* Fixed-focus by nature, meaning focus drift cannot occur and intrinsics remain fixed. However, distortion is more complex and low-light quality is lower.


* *Wide/Main & Telephoto Lenses:* Higher optical quality, but subject to depth-of-field variations and focus shifts.


* *Tip:* If using wide or telephoto lenses, **lock the focus distance** (e.g., enable autofocus on the subject, then immediately toggle autofocus off to lock the distance) and maintain a consistent physical distance from the subject. If shooting at variable distances, shoot separate batches and group cameras in the solver.




* **Exposure Triangle & White Balance:**
* *Shutter Speed:* Set to a fixed speed **$\ge$ 1/500s**. *Logic:* Prevents motion blur during walking/orbiting, which is a primary cause of reconstruction failure.


* *ISO:* Set to the **minimum available value**. *Logic:* Minimizes sensor noise and grain, ensuring clean feature matching.


* *White Balance:* Adjust manually until colors match the scene, then **lock it**. *Logic:* Prevents frame-to-frame color temperature shifts that create cloudiness or visual artifacts in the radiance field.




* **Shooting Path & Geometry:**
* *Angle:* Keep the camera pointed at an **oblique downward angle** toward the target.


* *Logic:* Minimizes excessive background and distant sky, saving compute time and eliminating floating artifacts.


* *Orbiting:* Move smoothly and slowly around the subject, ensuring complete coverage. You do not need to fit the entire object in every shot, but you must capture the subject from the prospective viewing angles expected in the final presentation.





---

**Phase 2: Model Generation (Mip Map Software)**

* **Frame Extraction:**
* *Tip:* Import video into Mip Map and set extraction intervals.


* *Alternative:* If significant motion blur is present in the recorded video, pre-process via *Sharp Frame* (with H.264 files) to automatically select clear keyframes.




* **Initial Focal Length Seeding:**
* *Tip:* Input the 35mm equivalent focal length (e.g., 65mm for the telephoto test) and use the built-in calculator to convert it to pixel focal length prior to reconstruction.


* *Logic:* Providing an accurate initial focal length prior accelerates camera pose recovery and prevents convergence errors during bundle adjustment.




* **Output Optimization:**
* *Tip:* Select "Ultra-High" quality, enable the **Gaussian Splatting** toggle, and disable 3D Mesh / Point Cloud generation if they are not needed.


* *Logic:* Saves computation time by computing only the desired radiance output.





---

**Phase 3: Cleanup & Post-Processing (Super Splat)**

* **Coordinate System Realignment:**
* *Tip:* Set the Z-axis rotation to zero and adjust the X-axis slider visually to align the ground plane and level the horizon.


* *Logic:* Straightens the scene for intuitive rotation, orbiting, and predictable keyframe camera positioning.




* **Selection Modes (Center vs. Ring):**
* *Center Mode:* Penetrating selection that grabs all splats within the 2D bounding area regardless of depth.


* *Ring Mode:* Surface-only selection that selects only the closest visible ellipsoids, protecting occluded geometry behind them.




* **Geometric Pruning Tools:**
* *Rectangular / Lasso / Polygon:* Standard boundary selection tools for quick bulk deletions.


* *Brush Select:* Allows manual painting of selections (resizable via bracket keys `[` / `]`).


* *Flood Select:* Uses connectivity and spatial density to isolate and remove floating artifacts.


* *Sphere Select + Invert:* Places a 3D bounding sphere around the target object, inverts the selection, and deletes all peripheral environment data.




* **Histogram-Based Filtering:**
* *Tip:* Open the Splat Data panel, choose statistical metrics, and use the **Log Scale** toggle to visualize dense distributions.


* *Logic:* Allows global filtering and deletion of low-confidence, extreme-scale, or stray splats based on numerical attributes rather than manual lassoing.




* **Keyframing & Export:**
* *Tip:* Set sequential viewing angles as keyframes to establish a clean camera trajectory for final video rendering or interactive sharing.