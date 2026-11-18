# ⭐ Five Star Clipper

**A free tool that finds the best moments in streams and turns them into ready-to-post videos.**

Paste a link from **Twitch, YouTube or Kick**. When chat goes crazy or the
streamer loses it, it saves the moment and makes:

- 📱 **Shorts** (for TikTok, YouTube Shorts, Reels): streamer's face on top,
  gameplay below, big captions.
- 🎬 **Long videos** (10–15 minutes for YouTube): the best moments of one or
  more streams, with a narrator, chapters and a thumbnail.

It runs on your own computer. No account, no sign-up, no payment.

---

## How to start

### Step 1: Download it

Click the green **Code** button at the top of this page, then **Download ZIP**.
Unzip it (double-click the ZIP file).

### Step 2: Double-click the icon for your computer

| Your computer | Double-click this |
|---|---|
| 🍎 **Mac** | **Five Star Clipper (Mac)** |
| 🪟 **Windows** | **Five Star Clipper (Windows)** |

**The first time only:**

- **Your computer will warn you** because the app isn't from an app store.
  - **Mac:** click **Done**, open **System Settings → Privacy & Security**,
    scroll down and click **Open Anyway**.
  - **Windows:** click **More info → Run anyway**.
- **A setup window opens and installs everything by itself.** It takes about
  10 minutes and needs about 3 GB of free space. Leave it open until it's done.
  - **Mac only:** if it says you need **Homebrew**, copy the line it shows,
    paste it in the same window, press Enter, then double-click the icon again.

After that, the clipper opens in your web browser, and a **Five Star Clipper**
shortcut appears on your Desktop. Next time, just double-click it.

---

## How to use it

**Make shorts**

1. Paste a stream link in the box and press **GO**.
   - A **live** stream: it watches and clips as things happen.
   - A **past** stream or video: it scans the whole thing and clips the best parts.
2. Your shorts appear under **Your Clips**. Press **Download**.
3. Press **See more** under a clip to get a title, description and hashtags
   you can copy.

**Make a long video**

1. Click the **🎬 Long videos** tab at the top.
2. Paste one or more past streams, type what the video is about, and press
   **BUILD STORYBOARD**.
3. Check the plan: rename chapters, change what the narrator says, or switch
   moments on and off.
4. Press **Render video**. In about 20 minutes you get the video, a thumbnail,
   titles and a description with chapters.

**Stop it:** press **⏻ Stop clipper** at the top of the page. Closing the
browser tab does *not* stop it, so it keeps clipping in the background.

---

## Good to know

- **Where your videos are saved:** in the `tool/clips` folder.
- **Background music for long videos:** put songs in `tool/music`. Only use
  music you're allowed to use (for example the free YouTube Audio Library).
- **Settings** (at the bottom of the page): start with your computer,
  notifications for big clips, phone alerts, auto-cleanup when your disk is full.
- **Moved the folder?** The icons in the folder still work. The Desktop
  shortcut won't: delete it, then double-click the icon in the folder again.
- **Something not working?** Look in `tool/clipper.log`, which says what went
  wrong. You can open an issue on GitHub and paste what it says.
- **Windows support is new.** If you hit a problem on Windows, please report it.

## Please clip fairly

Give credit to the streamers (the descriptions it writes already do), follow
each platform's rules, and respect streamers who don't want to be clipped.

---

More details on how it works: [tool/DETAILS.md](tool/DETAILS.md)

Free and open source under the [MIT License](LICENSE). Uses free tools and AI
models: ffmpeg, whisper.cpp, yt-dlp, streamlink, OpenCV, YAMNet and FER+
(full list in DETAILS.md).
