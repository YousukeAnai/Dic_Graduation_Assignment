# アンパンマン画伯判別機

![anpanman_anomally_detection_gaiyou](https://user-images.githubusercontent.com/46349770/70225769-01707500-1793-11ea-824f-4bf270d010b4.png)

## 実行手順

1.  下記を実行し、本リポジトリをクローン  
      !git clone git@github.com:YousukeAnai/Dic_Graduation_Assignment.git

2.  Trainデータを解凍。  
      !unzip -d "./ANPANMAN_Anomaly_Detection/Train_Data/191103/" "./ANPANMAN_Anomaly_Detection/Train_Data/191103/anpanman_train.zip"  
      !rm "./ANPANMAN_Anomaly_Detection/Train_Data/191103/anpanman_train.zip"

3.  実行ディレクトリに移動。  
      !cd "./ANPANMAN_Anomaly_Detection/EfficientGAN"

4.  学習を実行。(GoogleColabにて約8h)  
      !python train_BiGAN.py

5.  判定したいアンパンマン画像ファイルを下記ディレクトリに入れる。  
      ./ANPANMAN_Anomaly_Detection/Test_Data/191205/  
      (既に上記ディレクトリ内に入っている画像は参考画像なので削除)

6.  推定を実行  
      !python predict_BiGAN.py

## 判定結果  

    下記ファイルに表示される。  
    ./ANPANMAN_Anomaly_Detection/EfficientGAN/out_images_BiGAN/resultImage_anpanman_test.png
![result_image](https://user-images.githubusercontent.com/46349770/70229068-d7ba4c80-1798-11ea-85be-ad4a95a23a5b.png)
