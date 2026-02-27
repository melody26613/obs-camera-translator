from pygrabber.dshow_graph import FilterGraph

# 列出所有攝影機裝置
devices = FilterGraph().get_input_devices()

for i, device in enumerate(devices):
    print(i, device)

"""
output
0 Chicony USB2.0 Camera
1 OBS Virtual Camera
"""