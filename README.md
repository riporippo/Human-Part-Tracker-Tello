# yolo-poseモデルを用いたRyze Telloの自律接近
### 背景   
* 安価な構成でドローンの自律制御をやってみたい
* 自律追跡は簡単にできたけど，せっかくだしもっといろいろやってみたい
* 姿勢推定モデル使って、接近する部位を指定出来たら面白そう

### 機材構成
* Raspberry Pi 5
* Hailo-8
* Ryze Tello
* ホストとなるPC
* Raspberry Pi 5用のネットワークアダプタ

### 動作例
マネキンを対象にした実験を実施  
実験: 右肩への接近　　
* 人物の検出自体はできているが、keypointの検出が時々危うい　　
![output35](https://github.com/user-attachments/assets/b10c5a8f-ccd9-44c5-a627-af0b42dc5046)　　
![output38](https://github.com/user-attachments/assets/edf72a64-8660-40a7-97c5-c9ded6edbba6)  
* 画面内に映る人物の面積が一定になるよう前後を移動するため、keypointより少し上に飛んでしまう挙動を確認
![output79](https://github.com/user-attachments/assets/c4679530-bc77-4678-9c3a-d50f3ac62ec0)
### ※注意
**目的**: 本プログラムは個人的な研究・実験用です。  
**禁止事項**: 危険な改造や、公序良俗に反する目的での使用は行わないでください。  
**自己責任**: ドローンの回転体は非常に危険です。使用中の事故や怪我について、作者は一切の補償・関与をいたしません。本リポジトリの内容を参考にする場合は、必ず安全な環境で、自己責任の下で実施してください。
