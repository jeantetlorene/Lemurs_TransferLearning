# An integrated passive acoustic monitoring and deep learning pipeline applied to black-and-white ruffed lemurs (Varecia variegata) in Ranomafana National Park,Madagascar

## Purpose of the project

Fieldwork was conducted at Mangevo (21.3833S, 47.4667E), an isolated and undisturbed forest location within Ranomafana National Park (RNP), located in southeastern Madagascar, during the period of May to July 2019. To facilitate passive acoustic monitoring, we deployed a total of two SongMeter SM4 devices (manufactured by Wildlife Acoustics) and two Swift units (provided by the Cornell Yang Center for Conservation Bioacoustics). The placement of these recorders was strategically chosen within the central regions of known subgroups, ensuring a minimum distance of 300 meters between each device. The SongMeter devices operated at a sampling rate of 48 kHz, while the Swift units operated at 32 kHz, respectively, enabling comprehensive audio data collection throughout the study period.

## Authors

Carly H. Batist [1],[2], Emmanuel Dufourq [3],[4],[5], Lorene Jeantet [3],[4],[5], Mendrika N.Razafindraibe [6], Francois Randriamanantena [7], and Andrea L. Baden [1],[8]

[1]: The Graduate Center of the City University of New York, Department of Anthropology, New York, USA  
[2]: Rainforest Connection (RFCx), Katy, USA  
[3]: African Institute for Mathematical Sciences, South Africa  
[4]: Stellenbosch University, Department of Applied Mathematics, South Africa  
[5]: National Institute for Theoretical and Computational Sciences, South Africa  
[6]: University of Antananarivo, Department of Animal Biology, Antananarivo, Madagascar  
[7]: Centre ValBio, Ranomafana, Madagascar  
[8]: Hunter College of the City University of New York, Department of Anthropology, New York, USA  

## Demo

A quick 3 minute demo is available here on <a href="https://colab.research.google.com/drive/1G_zicIHNTrBJuiXJYsKqRMWktazdX3vx?usp=sharing">Google Colab demo</a>. The script applies the model to one audio file and generated predictions. The audio file and predictions can be loaded into Sonic Visualiser to that the predictions can easily be verified as shown:

![image](https://drive.google.com/uc?export=view&id=11ZA6GRCQcCJD6f7kFc3_EybuWzdiOUQ1)

A very faint black-and-white ruffed lemur vocalisation can be seen in the spectrogram above. The model is able to detect calls that were recorded far away from the microphone.

## Open source data

<a href="https://doi.org/10.5281/zenodo.7956064">Our training, testing and model is open source and can be accessed here. </a> We provide the audio data (.wav) used to train and test our neural network classifier along with the corresponding labelled text files (.data). Tensorflow model weights are provided. This dataset has 56 testing audio files (roughly 38 hours, 8.7GB) and 246 training files.

DOI for data: 10.5281/zenodo.7956064



## Description of the codes 
### Pre-processing of the data 

### Training of the model 

### Application of the model

