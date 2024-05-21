import cv2  # OpenCV kütüphanesini içe aktarıyoruz.
import matplotlib.pyplot as plt  # Matplotlib kütüphanesini içe aktarıyoruz.

# Şablon eşleme
img = cv2.imread("cat.jpg", 0)  # "cat.jpg" resmini gri tonlamalı olarak yüklüyoruz.
print(img.shape)  # Resmin boyutlarını yazdırıyoruz.
template = cv2.imread("cat_face.jpg", 0)  # "cat_face.jpg" resmini gri tonlamalı olarak yüklüyoruz.
print(template.shape)  # Şablon resminin boyutlarını yazdırıyoruz.
h, w = template.shape  # Şablonun yüksekliğini ve genişliğini alıyoruz.

# Kullanılacak şablon eşleme yöntemlerinin listesi
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Her bir yöntemi döngüde işliyoruz
for meth in methods:
    method = eval(meth)  # Metin olarak yazılmış yöntemi OpenCV fonksiyonuna dönüştürüyoruz.
    res = cv2.matchTemplate(img, template, method)  # Şablon eşleme işlemi yapılıyor.
    print(res.shape)  # Eşleşme sonucunun boyutlarını yazdırıyoruz.
    
    # Eşleşme sonucunun minimum ve maksimum değerlerini ve konumlarını buluyoruz.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Yönteme bağlı olarak en iyi eşleşme noktasını belirliyoruz.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc  # TM_SQDIFF ve TM_SQDIFF_NORMED yöntemlerinde en düşük değeri kullanıyoruz.
    else:
        top_left = max_loc  # Diğer yöntemlerde en yüksek değeri kullanıyoruz.
    
    # Eşleşme bölgesinin sağ alt köşesini hesaplıyoruz.
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Eşleşme bölgesine bir dikdörtgen çiziyoruz.
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    # Sonuçları görselleştiriyoruz.
    plt.figure()  # Yeni bir figür oluşturuyoruz.
    plt.subplot(121), plt.imshow(res, cmap="gray")  # Eşleşme sonucunu görüntülüyoruz.
    plt.title("Eşleşen Sonuç"), plt.axis("off")  # Başlık ve eksenleri kapatıyoruz.
    plt.subplot(122), plt.imshow(img, cmap="gray")  # Orijinal resmi eşleşme dikdörtgeni ile görüntülüyoruz.
    plt.title("Tespit edilen Sonuç"), plt.axis("off")  # Başlık ve eksenleri kapatıyoruz.
    plt.suptitle(meth)  # Yöntemin adını üst başlık olarak ekliyoruz.

    plt.show()  # Grafiği gösteriyoruz.