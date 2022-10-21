from __future__ import unicode_literals
import youtube_dl

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

ydl_opts = {
    'download_archive': 'archive.txt',
    'ignoreerrors': True,
    'nooverwrites': True,
    'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
    'outtmpl': '저장 경로 템플릿',
    'noplaylist' : False,
    'progress_hooks': [my_hook],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['유튜브 링크 주소'])



