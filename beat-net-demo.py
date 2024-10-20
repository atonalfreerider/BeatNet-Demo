from BeatNet.BeatNet import BeatNet

estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

Output = estimator.process("/home/john/Desktop/Audio/BeatNet-Demo/william-larissa.mp3")

