# GSoC' 19

### Summary of Project
The project initially sets out to achieve two goals – update the annotation system for Red Hen’s NewsScape dataset to FrameNet 1.7 and expand FrameNet 1.7 through a knowledge-driven approach and a distributional semantics approach. 

I achieved the first objective by updating the annotation system for Red Hen’s NewsScape dataset to FrameNet 1.7 using PyDaisy and Open-Sesame parsers. However, there are two important changes. 

First, I did not use `pyfn` library as suggested in my proposal because the library is not intended for annotating sentences outside of FrameNet and I could not resolve the bugs when deploying the library despite opening issues and working with the library's creator through GitHub. I changed to use PyDaisy and OpenSesame standalone library. PyDaisy is the alternative to SimpleFrameID for frame identification, and the library OpenSesame is also capable of identifying target words and frames. 

Second, SEMAFOR could not be implemented because `pyfn` was not working and the pretrained models (MaltParser trained on Penn Treebank and the model files for SEMAFOR trained on the FrameNet 1.7 datasets) in https://github.com/AlenUbuntu/semafor_Framenet_v1.7 were missing. I had sent multiple follow-up emails to the author and the person-of-contact listed in the repository but I had not received any reply to date.

**Documentation of PyDaisy and Open-Sesame parsers** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/Documentation%20-%20PyDaisy%20and%20OpenSesame.md
**Detailed reasons of not using `pyfn`** - https://github.com/yongzx/GSoC-2019-FrameNet/blob/master/GSoC%20Phase%201_%20Report%20on%20pyfn.pdf

