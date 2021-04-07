# Captcha-Recognizer
A light `CNN` + `Attention` based Captcha Recognizer  
---  
## Get Started  
### Install Dependencies
- Dependencies  
`torch`, `torchvision`, `tqdm`, `pillow`, `numpy`

### Ready Train Dataset 
- Captcha image
  ```  
  image_path
    ├─ 1.jpg
    ├─ 2.jpg
    ┗─ ...
  ```
- label_path (`.txt`)
  ```
  1.jpg\t1234\n
  2.jpg\t5678\n
  ...
  ```  
### Run  
```  
python train.py
```

Enjoy the code :)
