## Progression Boost

### Introduction

Thanks to Ironclad and their grav1an, Miss Moonlight and their Lav1e, and Trix and their autoboost that makes this script possible.

Progression boost is a boosting script for maintaining a consistent quality throughout the whole encode. It works in the same way as Ironclad's grav1an using polynomial regression, and can be easily configured to target different quality levels and more.  

### Usage

Progression Boost is very customisable. For this reason, we offers multiple presets with different quality targets. You can choose one that's closer to what you want to achieve with your encode, and then modify the config from that.  

| Preset | Quality Target Explained |
| :-- | :-- |
| [Butteraugli-3Norm-90th-Percentile](Progression-Boost/Preset-Butteraugli-3Norm-90th-Percentile/Progression-Boost.py) | Targeting very high quality, focusing on getting even the worst frame good. |
| [SSIMU2-20th-Percentile](Progression-Boost/Progression-Boost.py) | Targeting high to very high quality, focusing on getting the bad frames good. |
| [SSIMU2-Harmonic-Mean](Progression-Boost/Preset-SSIMU2-Harmonic-Mean/Progression-Boost.py) | Targeting from low to high quality, focusing on quality consistency. |
| [Butteraugli-3Norm-Root-Mean-Cube](Progression-Boost/Preset-Butteraugli-3Norm-Root-Mean-Cube/Progression-Boost.py) | Targeting medium-high to high quality, focusing on quality consistency. |

After you've downloaded a preset, open the file in a text editor and follow the guide at the very top to adjust the config. The guide should be able to lead you through all the options for Progression Boost.

Also, remember to install the dependencies for Progression Boost at [`requirements.txt`](Progression-Boost/requirements.txt) using pip.

### Note

* Progression Boost will encode the video multiple times until it can build a polynomial model. If you prefer a faster option that only encodes the video once and boost using a „magic number“, try Miss Moonlight's Lav1e or Trix's autoboost.  

## Dispatch-Server

### Introduction

This dispatch server aims to solve two common problems in AV1 encoding:  

1. There has been rigorous testing by BONES and myself regarding which `--workers` and `--lp` combo to use.  
In general, for the commonly distributed SVT-AV1-PSY / SVT-AV1-PSYEX binaries, something around `--lp 3` should be preferred. However, for `--workers`, since the amount of resource SVT-AV1-PSY / SVT-AV1-PSYEX uses varies when encoding different scenes, setting a fix amount of `--workers` will always result in the whole Av1an sometimes overloading and sometimes underutilising the system.  
This dispatch server mitigates this by monitoring the CPU usage and only dispatching a new worker when there's free CPU available.  

2. For heavily filtered encodes, it's very easy to run into VRAM limitations as Av1an runs multiple workers in parallel. It's suboptimal to use exactly the amount of workers that would fully utilise the VRAM, because when VSPipe for a worker has finished filtering but the encoder is still yet to consume the lookahead frames, the worker is not using the VRAM. It's also not possible to use slightly more workers than the VRAM allows, because by chance sometimes VSPipe for all workers will run in the same time, and the system would freeze and potentially completely halt.  
This dispatch server solves this by monitoring the VRAM usage and only dispatching a new worker when there's free VRAM available.  

### Usage

The dispatch server consists of three parts:  

* [`Server.py`](Dispatch-Server/Server.py): This is the main script for the dispatch server. All the monitoring and dispatching happen in this script.  
* [`Server-Shutdown.py`](Dispatch-Server/Server-Shutdown.py): This is the shutdown script for `Server.py`. This shutdown script can be automatically run after encoding finished.  
* [`Worker.py`](Dispatch-Server/Worker.py): The lines of codes in this script will need to be copied to the top of the filtering vpy script. It halts the execution of the vpy script until it receives the green light from the dispatch server.  

To adapt the dispatch server:  

1. Check the [`requirements.txt`](Dispatch-Server/requirements.txt) in the folder. This `requirements.txt` can directly be used for NVIDIA GPUs. For other GPU brands, replace the `nvidia-ml-py` package in the `requirements.txt` with the appropriate package. After that, use pip to install the dependencies for the dispatch server from `requirements.txt`. Running the dispatch server in the same Python as the Python used for filtering is recommended.  
2. Download the [`Server.py`](Dispatch-Server/Server.py) and [`Server-Shutdown.py`](Dispatch-Server/Server-Shutdown.py). Open `Server.py` in a text editor, and at the top there will be several variables configuring the amount of VRAM and CPU usage expected for each worker, among other settings. Follow the guides in the file to adjust all the variables. For non-NVIDIA GPUs, replace `pyvnml` with appropriate monitoring tool to continue.  
3. Copy everything in [`Worker.py`](Dispatch-Server/Worker.py) and follow guide in the file to paste it into the filtering vpy script.  

To use the dispatch server:  

1. Run `Server.py` in the background or in a different terminal.  
2. Run Av1an using the modified filtering vpy script that includs the lines from `Workers.py`. For Av1an parameter `--workers`, set an arbitrarily large number of workers for Av1an to spawn so that the dispatch server will always have workers to dispatch when there's free CPU and VRAM.  
3. After encoding, either run `Server-Shutdown.py` to shutdown the server, or Crtl-C or SIGKILL the server process.  

### Note

* Windows' builtin Task Manager is not a good tool for checking CPU usage. The CPU Utility reported in Task Manager will never reach 100% on most systems, despite the CPU is already delivering all the performance it can. This is not an advertisement, but HWiNFO, a tool commonly used by PC building community, shows a different CPU Usage number, which is more aligned to what people expects.  
