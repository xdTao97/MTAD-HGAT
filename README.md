# Anomaly Detection for Multivariate Time Series Data Based on Multi-Head Graph Attention


## Datasets
Dataset download address:
    wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

    cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
Download the data set and import it into the project. The effect is shown as follows:
![dataset](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/dataset.png)
## Installation
```
1. Install python.
2. Use git command to download the project to local.
3. Install condconda.
4. Install the corresponding operating environment according to environments.txt.
```

## Running code
* 1. First run preprocess.py to process the data set. Taking the SMAP data set as an example, the running results are as follows:
![datasetProcess](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/datasetProcess.png)
* 2. Run main.py again to get the model running results.
![output_example](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/output_example.png)

## Notice
We can set parameters such as data set, multi-head graph attention and sliding window through the configuration file args.py, and the final running results will be saved in the output folder.
As shown below:
![result](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/output.png)


## Experiment
The model has been determined, and all experiments can be performed by modifying the parameters in the configuration file args.py.
* (1) Basic experimental results can be obtained by modifying args.dataset. The two sets of data sets were subjected to ten experiments respectively, and the average value was calculated to obtain the final result. Taking the SMAP data set as an example, the obtained results are shown in Figure ex_result：![ex_result](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/ex_result.png)
The bf_result in the figure is the final result we used. Statistics on f1, precision, and recall get the TABLE II experimental results. Statistics on the AUC indicator get the experimental results on AUC in Fig4 and Fig5.
* (2) Statistics and calculation of epsilon_result and pot_result in summary.txt to obtain the experimental results of TABLE IV.
* (3) Modify num_heads in the args.py file to 1,2,3,4 to get the experimental results of Fig6.
* (4) Modify the lookback value in the args.py file and set it to 30, 60, 60, 110, 130 to get the experimental results of Fig7.
* (5) In TABLE III, we conducted Ablation Study. By annotating the corresponding modules in the mtad_gat.py file, the final experimental results are obtained. For the annotation module, please refer to the image_annotation image below:
![image_annotation](https://github.com/xdTao97/timeSeriesAnomalyDetect--MTAD-HGAT/blob/master/fig/mtad_gat.png).
At the same time, modify the code in the aggregation class in modules.py to be as follows:
```
    '''
    #Use three parameters: x, h_feat, h_temp
    def forward(self, x1, x2, x3):
        x1_1 = x1
        x1_2 = x2
        x1_3 = x3

        x2_1 = torch.cat((x1_1, x1_2), 2)
        x2_1 = self.conv_concat2(x2_1)

        x2_2 = torch.cat((x1_1, x1_3), 2)
        x2_2 = self.conv_concat3(x2_2)

        x3_1 = torch.cat((x2_1, x2_2), 2)
        x3_1 = self.conv5(x3_1)
        x3_1 = x3_1.transpose(1, 2)
        return x3_1
    '''
    #Use only h_temp or only h_feat in ablation experiments
    def forward(self, x1, x2):
        x1_1 = x1
        x1_2 = x2

        x2_1 = torch.cat((x1_1, x1_2), 2)
        x2_1 = self.conv_concat2(x2_1)

        x3_1 = self.conv5(x2_1)
        x3_1 = x3_1.transpose(1, 2)
        return x3_1
```
## Email
If you have any questions, you can contact us by email. Email：taoxiaodong163@163.com      
