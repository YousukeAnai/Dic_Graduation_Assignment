# アンパンマン画伯判別機

\![anpanman_GAHAKU](https://user-images.githubusercontent.com/46349770/66885475-5ef80900-f00f-11e9-9927-28f27503b60c.png)    
![anpanman_anomally_detection_gaiyou](https://user-images.githubusercontent.com/46349770/70225769-01707500-1793-11ea-824f-4bf270d010b4.png)

##実行手順

1.  GoogleDriveにて任意の実行ノートブックを立ち上げる。

2.  下記を実行し、本リポジトリをクローン  
      !git clone git@github.com:YousukeAnai/Dic_Graduation_Assignment.git

3.  実行ディレクトリに移動。  
      !cd "./ANPANMAN_Anomaly_Detection/EfficientGAN"

4.  学習を実行。(約8h)  
      !python train_BiGAN.py

5.  判定したいアンパンマン画像ファイルをディレクトリTest_Data/に置く。

6.  推定を実行  
      !python predict_BiGAN.py

##判定結果  

    下記ファイルに表示される。  
    ./ANPANMAN_Anomaly_Detection/EfficientGAN/out_images_BiGAN/resultImage_anpanman_test.png
![result_image](https://user-images.githubusercontent.com/46349770/70229068-d7ba4c80-1798-11ea-85be-ad4a95a23a5b.png)
