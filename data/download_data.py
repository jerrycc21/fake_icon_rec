import os
import json 
import multiprocessing
from functools import partial

urls = json.load(open("./icon_url.json"))
p = multiprocessing.Pool(5)

def download(url):
    cmd = "wget %s -P icons/" % url
    os.system(cmd)
p.map(download, urls)
p.close()
p.join()
