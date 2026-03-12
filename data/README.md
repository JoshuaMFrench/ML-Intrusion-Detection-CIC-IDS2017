# Data

This folder should contain the CIC-IDS2017 dataset CSV files. The dataset is not included in this repository due to its size (2.8GB+).

---

## Download Instructions

1. Go to the [Canadian Institute for Cybersecurity dataset page](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Download the **MachineLearningCSV.zip** file
3. Extract the contents
4. Place all CSV files directly into this `/data` folder

---

## Expected Files

After extraction your `/data` folder should contain these files:

```
data/
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

Once the files are in place, run the pipeline from the root directory:

```bash
python ids_pipeline.py
```
