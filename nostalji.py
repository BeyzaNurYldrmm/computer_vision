
import cv2 as cv
import numpy as np
import gradio as gr
"""
#imread fonk. görseli okur ve sayı dizisi şeklinde alır
img=cv.imread(r"D:\goruntu_isleme\satranc3x3.jpg")
#print(img)#sayı dizisine çevirilecek
#opencv renk paleti bgr
print(img[300:40 0])
#görsel olarak göster
cv.imshow("gorsel",img)#türkçe karakter kullanmayın
#bekle klavyeden tusa basılınca cık
cv.waitKey(0)
cv.destroyAllWindows()"""

resim=r'D:\\goruntu_isleme\\yesil_elma.jpg'
img = cv.imread(resim)  # Dosya yolundaki ters çizgileri çift yazarak kaçırıyoruz

# Görüntü boyutlarını alma
print(img.shape)
height, width = img.shape[:2]
print(height, width)
print(f'Yükseklik: {height}')
print(f"Veri tipi: {img.dtype}")

# Farklı okuma modları
img_color = cv.imread(resim, cv.IMREAD_COLOR)
#img_color = cv.imread(resim, 1)
img_gray = cv.imread(resim, cv.IMREAD_GRAYSCALE)
#img_gray = cv.imread(resim, 0)

# Görüntüleri gösterme
cv.imshow("Renkli", img_color)
cv.imshow("Gri", img_gray)
cv.waitKey(0)
cv.destroyAllWindows()


#*************----------********************
#gradio web tabanlı aratüz tasarlamak için
#   ----------**********--------------

def nostalji(image):
    image = np.array(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray

# Gradio ara yüzünü oluşturma
with gr.Blocks() as demo:
    gr.Markdown("### Resmi Siyah-Beyaz Çevirici")
    gr.Markdown("Bir resim yükleyin ve siyah beyaz nostaljiye çevirin!")
    
    image_input = gr.Image(type='pil', label="Giriş Resmi")
    image_output = gr.Image(type='numpy', label="Sonuç Resmi")
    
    # Fonksiyon bağlantısı ve çalıştırma
    demo.interface = gr.Interface(fn=nostalji, inputs=image_input, outputs=image_output)
    
# Gradio arayüzünü başlatma
if __name__ == "__main__":
    #demo.launch()
    demo.launch(share=True)#link paylaş 72 saat sonra link süresi dolar
    



