README

Inkluderar följande källkodsfiler:
Matlab:
build_supervised_dataset_fake_friction.m
logistic_regression.m
build_supervised_dataset.m
run_svm.m
Python:
TFANN.py

Resursfiler:
dataset_list.xlsx
query_weatherstation_save_1435.csv, SMHITemp.csv - väderleksdata

**Fil 1: build_supervised_dataset_fake_friction.m
Denna fil används för att bygga ett data set med fejkade friktionsvärden i kombination med 
uppmätt väderleksdata. Friktionsvärderna viktas med avseende på fuktighet och temperatur 
och en slumpfaktor:

friktionsvärde = (temperatur+fuktighet+slump)/3

där slump variabeln varierar mellan 0 och 0.5 (se rad 329).

De tre fejkade föregående friktionsvärderna skapas enligt:

föregående_friktionsvärde1 = (friktionsvärde+fuktighet_från_30min+temperatur_från_30min)/3
föregående_friktionsvärde2 = (friktionsvärde+fuktighet_från_60min+temperatur_från_60min)/3
föregående_friktionsvärde3 = (friktionsvärde+fuktighet_från_90min+temperatur_från_90min)/3

Där fuktighet och temperatur tas från 30, 60 samt 90 minuter sedan (se rad 348,370,390). Avstånden är 
satta till 0.005, 0.01 samt 0.02 med måttkvalite 5. Fuktigheten och temperaturen har en negativ
korrelation till friktionsvärdet. Därför tar vi det negativa värdet och begränsar detta mellan
noll och ett (se kod).

Slutresultatet sparas i variabeln 'cleareddataset' och expoteras som cleareddataset.mat och cleareddataset.csv.

**Fil 2: logistic_regression.m
Bygger Logistic regression modeller

**Fil 3: run_svm.m
Bygger SVM modeller

**Fil 4: build_supervised_dataset.m
Bygger data set från riktiga friktionsmätningar och väderleksdata i likehet med 
build_supervised_dataset_fake_friction.m.

**Fil 5: TFANN.py
Bygger ANN modeller
För att köra TFANN.py behövs python 3 och följande paket:
scipy.io
tensorflow (https://www.tensorflow.org/get_started/os_setup)
numpy
pandas
sklearn


**Fil 6: dataset_list.xlsx
Beskriver datastrukturen som skapas av build_supervised_dataset_fake_friction.m
samt build_supervised_dataset.m.
