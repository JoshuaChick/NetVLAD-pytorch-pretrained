# Run
```
git clone https://github.com/JoshuaChick/NetVLAD-pytorch-pretrained
```
```
cd NetVLAD-pytroch-pretrained
```
```
pip3 install -r requirements.txt
```
```
python script.py
```
Notes: 
- Put your image paths in ```img_tensor1 = preprocess_image('<image path 1>')``` and ```img_tensor2 = preprocess_image('<image path 2>')``` in ```script.py``` to run on your own images.
- ```script.py``` will give you the distance between the NetVLAD vectors for two images (between 0 and 2, as vectors each have magnitude 1).
- ```script.py``` compares two images separately for clarity. However, you can pass in a bacth of images in ```image_tensor``` for ```extract_features(mdl, image_tensor)```. The batch must be in the form [batch_size, channels, h, w].