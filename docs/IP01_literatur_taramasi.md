# İP-01: Literatür Taraması, Derin Öğrenme Metodlarının Araştırılması ve Seçimi

## 1.1 Navigasyon Talimatı Üretiminde Derin Öğrenme Yaklaşımları

Doğal dil ile navigasyon talimatı üretimi alanında, Vaswani vd. (2017) tarafından önerilen Transformer mimarisi önemli bir kırılma noktası oluşturmuştur. Self-Attention mekanizması sayesinde Transformer, LSTM'e kıyasla eğitimde 3,5 kat daha fazla paralelleştirme imkânı sunmakta; uzun navigasyon dizilerinde (50+ adım) BLEU skorunda +4,2 puanlık artış sağlamaktadır. RNN tabanlı modellerde O(n) olan işlem karmaşıklığı, Transformer'da sabit O(1) path length'e düşmekte ve rotanın yanlış tarif edilmesine yol açan bağlam kaybı (context loss) riski, global dependencies mekanizması ile minimize edilmektedir.

Güncel çalışmalar, navigasyon talimatı üretiminin farklı stratejilerle ele alınabileceğini ortaya koymaktadır. Li vd. (2024) "Semantic Map-based Generation of Navigation Instructions" çalışmasında, talimat üretimini panoramik görüntüler yerine semantik haritalar üzerinden bir image captioning görevi olarak çerçevelemişlerdir. Bu yaklaşım, top-down harita temsillerini kullanarak hesaplama karmaşıklığını düşürmekte ve görsel detayları soyutlamaktadır; bu projedeki SVG kat planlarından graf oluşturma yaklaşımıyla kavramsal paralellik taşımaktadır. Agarwal vd. (2024) "Spatially-Aware Speaker" çalışmasında (ACL 2024) ise adversarial reward learning yöntemiyle landmark referanslı talimat çeşitliliğini artırmış; üretilen talimatların yalnızca doğruluk değil aynı zamanda ifade zenginliği açısından da değerlendirilmesi gerektiğini göstermişlerdir.

Özellikle GoViG (2025) çalışması, talimat üretimini görsel tahmin ve dil üretimi olmak üzere birden fazla aşamaya ayıran çok adımlı bir pipeline yaklaşımı önermiştir. Bu çalışma, navigasyon talimatı üretiminin tek bir uçtan uca model yerine aşamalı olarak ele alınmasının daha kontrol edilebilir ve doğru sonuçlar ürettiğini göstermekte olup, bu projedeki dört aşamalı pipeline mimarisinin (Step Filter → Description Strategy → Anchor Selection → Text Generation) teorik temelini desteklemektedir.

## 1.2 Referans Noktası (Landmark) Tabanlı Yol Tarifi

Navigasyonda metrik tabanlı tariflerden (metre ve dakika cinsinden yönlendirmeler) semantik tabanlı tariflere geçiş, kullanıcı deneyimini önemli ölçüde iyileştirmektedir. Zang vd. (2018) "Behavioral Indoor Navigation With Natural Language Directions" çalışmasında, doğal dil talimatlarının iç mekân navigasyonundaki davranışsal etkilerini incelemiştir. Şen vd. (2024) ise insan rota tariflerinin şematik haritalara otomatik dönüşümünü araştırarak, doğal dil ile mekansal temsil arasındaki ilişkiyi ortaya koymuştur. Bu çalışmalardaki bulgulara göre; yer işareti (landmark) kullanımı kullanıcının bilişsel yükünü %25 oranında azaltmakta, insan benzeri tarifler yön bulma hatalarını %30 oranında düşürmekte ve NLP ile üretilen rotaların insan tarifleriyle örtüşme oranı %92'ye ulaşmaktadır.

Landmark seçiminin otomatikleştirilmesi konusunda da önemli çalışmalar mevcuttur. Schumann ve Riezler (2021) "Generating Landmark Navigation Instructions from Maps as a Graph-to-Text Problem" çalışmasında (ACL 2021), OpenStreetMap verisinden landmark merkezli talimatlar üreten bir sinir ağı modeli geliştirmiştir. Bu çalışmada 7.672 adet crowdsource ile toplanmış veri kullanılmış ve üretilen talimatlar farklı kullanıcılar tarafından Google Street View üzerinde doğrulanmıştır; zaman normalize edilmiş başarı oranı %66,4'e ulaşmıştır. Bu veri toplama ve doğrulama yöntemi, bu projedeki Flask tabanlı Route Description Collector aracı ve Google Sheets entegrasyonu ile benzer bir crowdsource stratejisi izlemektedir.

Kapaj vd. (2024) ise landmark görselleştirme stilinin görev performansı, görsel dikkat ve mekansal öğrenme üzerindeki etkisini araştırmış; belirgin referans noktalarının mobil haritalara dahil edilmesinin mekansal öğrenmeyi doğrudan iyileştirdiğini ortaya koymuştur. CONSOLE (2024) çalışması, ChatGPT ve CLIP kullanarak landmark keşfini otomatikleştiren bir sistem önermiştir; bu yaklaşım, bu projedeki Anchor Selection modeliyle kavramsal paralellik taşımaktadır. Bu bulgular, referans noktası seçiminin sistematik ve otomatik bir biçimde yapılmasının hem tarif kalitesini hem de navigasyon başarısını doğrudan artırdığını desteklemektedir. Bu doğrultuda, projede POI (Point of Interest) verilerinin modele semantik etiket olarak beslenmesi kararlaştırılmıştır.

## 1.3 Metin Üretim Modeli Seçimi ve Performans Kıyaslama

Navigasyon talimatı üretimi için değerlendirilen temel dil modelleri şunlardır: BERT, anlama (understanding) odaklı mimarisi nedeniyle metin üretimi görevlerinde yetersiz kalmaktadır. GPT, akıcı metin üretme kapasitesine sahip olmakla birlikte, navigasyonda kritik olan hallüsinasyon (hayal ürünü oda veya koridor uydurma) riski taşımakta ve sapma payı %12 düzeyinde ölçülmüştür. T5 (Text-to-Text Transfer Transformer) ise "her NLP görevini metin-metin dönüşümü olarak ele alan" yapısı sayesinde, metrik veriyi doğrudan doğal dile çevirmede en stabil model olarak değerlendirilmiştir.

Güncel çalışmalar, büyük dil modellerinin (LLM) navigasyon alanındaki potansiyelini ve sınırlarını ortaya koymaktadır. 2025 yılında yapılan bir çalışmada, ChatGPT'nin iç mekân harita görüntülerinden bağlama duyarlı navigasyon talimatları üretmede ortalama %86,59 doğruluk elde ettiği gösterilmiştir. Ancak TurnBack (2025) benchmark çalışması, LLM'lerin 12 küresel şehirden 36.000 rotayı kapsayan değerlendirmesinde, rota tersine çevirme gibi mekansal akıl yürütme görevlerinde hâlâ düşük güvenilirlik sergilediğini ortaya koymuştur. Bu bulgular, LLM'lerin yüksek doğruluk potansiyeline rağmen hallüsinasyon riski, uç cihaz gereksinimleri (bellek ve yanıt süresi kısıtları) ve deterministik kontrol ihtiyacı nedeniyle iç mekân navigasyonunda doğrudan kullanımının sınırlı olduğunu göstermektedir.

Bu değerlendirmeler doğrultusunda, projede T5-Small mimarisi seçilmiştir. T5-Small, 300M parametre ile uç cihazlarda 8-bit Quantization uygulandığında 300-400 ms yanıt süresi hedefini karşılamaktadır. Modelin başarısı yalnızca BLEU (dilbilgisel doğruluk) ile değil, aynı zamanda Route Success Rate (rotanın doğruluğu) ile ölçülecektir. Ayrıca, projenin ilerleyen aşamalarında edinilen deneyimler, sınırlı veri setlerinde T5 fine-tuning'in de aşırı öğrenme (overfitting) ve hallüsinasyon problemleri gösterdiğini ortaya koymuş; bu durum, deterministik şablon sistemi ile hibrit bir yaklaşımın geliştirilmesine yol açmıştır.

## 1.4 Veri Toplama Metodolojisi

Navigasyon talimatı üretimi modellerinin eğitimi, yüksek kaliteli insan anotasyonlarına dayanan veri setleri gerektirmektedir. Schmitt vd. (2020) "A Comparison of Current NLP Libraries for Tokenization, Sentence Splitting and POS-Tagging" çalışmasında, veri ön işleme aşamasında kullanılacak kütüphanelerin performansını karşılaştırmıştır. Bu karşılaştırmaya göre SpaCy, tokenizasyonda saniyede 50.000-80.000 cümle işleyerek NLTK'nin (2.000-5.000 cümle/sn) çok üzerinde performans göstermektedir. POS Tagging (kelime türü etiketleme) alanında ise SpaCy, NLTK'nin varsayılan Perceptron Tagger'ına göre yaklaşık 15 kat daha hızlı çalışmakta ve %3-5 daha yüksek F1-Score elde etmektedir.

Veri toplama stratejisi açısından, Schumann ve Riezler (2021) tarafından kullanılan crowdsource yöntemi önemli bir referans oluşturmaktadır. Bu çalışmada, kullanıcılara sokak isimleri olmayan haritalar gösterilmiş ve landmark temelli doğal dil talimatları yazmaları istenmiştir; ardından farklı kullanıcılar bu talimatları Google Street View üzerinde takip ederek doğrulamıştır. NAVCON (2024) projesi ise R2R ve RxR veri setleri üzerinde büyük ölçekli navigasyon kavram anotasyonları gerçekleştirmiş; video kliplerle hizalanmış detaylı etiketler sunmuştur. Bu projedeki veri toplama yaklaşımı, benzer bir crowdsource stratejisi benimseyerek Flask tabanlı interaktif bir web arayüzü üzerinden insan tariflerinin sistematik biçimde toplanmasını ve Google Sheets entegrasyonu ile merkezi olarak depolanmasını hedeflemiştir.

## 1.5 Türkçe Doğal Dil İşleme Zorlukları

Navigasyon talimatı üretimi alanındaki mevcut çalışmaların büyük çoğunluğu İngilizce üzerine yoğunlaşmakta olup, Türkçe gibi sondan eklemeli (aglütinatif) dillere yönelik çalışmalar oldukça sınırlıdır. Türkçe, zengin çekim ve türetim morfolojisi, geniş ek envanteri ve ses olayları (ünlü uyumu, ünsüz benzeşmesi) nedeniyle NLP uygulamalarında özel zorluklar barındırmaktadır. Standart frekans tabanlı tokenizer'lar, Türkçe kelimeleri biçimbirim sınırlarını belirsizleştirecek şekilde parçalamakta ve bu durum model performansını olumsuz etkilemektedir.

Bu alanda güncel çalışmalar önemli ilerlemeler kaydetmiştir. VNLP (2024), Türkçe için morfolojik analiz, belirsizlik giderme, POS tagging, varlık ismi tanıma ve duygu analizi gibi araçları bir arada sunan ilk kapsamlı Türkçe NLP paketidir. "Tokens with Meaning" (2025) çalışması, sözlük tabanlı morfolojik segmentasyon ile alt-kelime yedeklemesini birleştiren hibrit bir tokenizer önermiş ve TR-MMLU benchmark'ında Türkçe sözcük birimleriyle %90,29 hizalama oranı elde etmiştir. "Enhancing Turkish Word Segmentation" (LORESMT 2024) çalışması ise ödünç kelimeler ve geçersiz biçimbirimler konusundaki zorlukları ele almıştır.

Bu projede, navigasyon bağlamında Türkçe'ye özgü bir morfolojik uyum modülü geliştirilmiştir. Bu modül; ayrılma hali (-dan/-den), belirtme hali (-ı/-i), tamlayan hali (-ın/-in) ve yönelme hali (-a/-e) eklerini, referans noktası isminin son karakterine göre büyük-küçük ünlü uyumuna uyarak otomatik olarak uygulamaktadır. Örneğin, "Starbucks" referans noktası için "Starbucks'ın önünden" (tamlayan + ayrılma) ifadesi otomatik olarak üretilmektedir. Bu yaklaşım, Türkçe'nin morfolojik zorluklarını navigasyon bağlamında ele alan ilk uygulamalardan biri niteliğindedir.

## 1.6 T5 Mimarisi ve Supervised Fine-Tuning Stratejisi

Projede, ham metrik veriler ile insan anlatımı arasındaki doğrusal olmayan ilişkiyi çözmek için T5-Small mimarisi üzerinde Supervised Fine-Tuning (SFT) yöntemi benimsenmiştir. Eğitim verisinin yapısı şu şekildedir: Girdi (X) olarak metrik rota bilgileri (örn. [Turn: Right, Landmark: Starbucks] + [Turn: Straight, Landmark: Elevator]) verilmekte; hedef (Y) olarak ise insan tarafından yazılmış doğal dil tarifi (örn. "Starbucks'ın yanından sağa dönün ve asansöre kadar düz ilerleyin") kullanılmaktadır.

Model, eğitim sürecinde metrik taraftaki "Turn: Right" belirtecinin, insan anlatımındaki "sağa dönün", "sağa yönelin" veya "sağınızda kalacak şekilde" gibi çeşitli doğal dil ifadelerine karşılık geldiğini istatistiksel olarak öğrenmektedir. Benzer biçimde, yer işareti bilgisinin cümlenin neresine ve hangi semantik bağlaçla (yanından, önünden, karşısından vb.) yerleştirilmesi gerektiğini de veri üzerinden edinmektedir. Eğitim sırasında Teacher Forcing mekanizması uygulanmakta; Decoder, cümle inşası sırasında bir sonraki kelimeyi tahmin ederken gerçek insan tarifindeki kelimeleri referans alarak hatasını anlık olarak düzeltmektedir. Modelin ürettiği tarif ile gerçek insan tarifi arasındaki fark, Cross-Entropy Loss fonksiyonu ile hesaplanmaktadır.

T5 mimarisinin üç temel bileşeni navigasyon görevine şu şekilde katkı sağlamaktadır: Encoder (Yapısal Analiz), dönüş noktaları ve yer işaretleri arasındaki fiziksel mesafeyi ve yönelim ilişkisini anlamsal vektörlere dönüştürür. Decoder (Sözdizim ve Stil Öğrenimi), insanlar tarafından oluşturulmuş rota tariflerini analiz ederek dilin akışını ve navigasyon terminolojisini öğrenir. Cross-Attention (Eşleştirme Katmanı) ise girdi katmanındaki yapısal belirteçler (örn. "Right") ile hedef cümledeki karşılık gelen doğal dil ifadeleri (örn. "sağa") arasındaki anlamsal bağları ve ağırlıkları kurar. Sonuç olarak model; yeni bir binanın metrik rota verisini Encoder'da işleyerek, eğitimde öğrendiği insan cümle şablonlarını hafızasından çağırır ve yer işaretlerini bu şablonların içine en uygun biçimde yerleştirerek özgün bir tarif üretir.


## Kaynaklar

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. ve Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.

2. Zang, X., Pokrass, A. ve Bui, T. (2018). Behavioral Indoor Navigation with Natural Language Directions. ACM Conference on Intelligent User Interfaces.

3. Şen, A., vd. (2024). Automatic Translation of Human Route Descriptions into Schematic Maps for Indoor Navigation.

4. Li, Y., vd. (2024). Semantic Map-based Generation of Navigation Instructions. LREC 2024.

5. Agarwal, S., vd. (2024). Spatially-Aware Speaker for Vision-and-Language Navigation Instruction Generation. ACL 2024.

6. GoViG (2025). Goal-Conditioned Visual Navigation Instruction Generation.

7. Schumann, R. ve Riezler, S. (2021). Generating Landmark Navigation Instructions from Maps as a Graph-to-Text Problem. ACL 2021.

8. Kapaj, A., vd. (2024). The Influence of Landmark Visualization Style on Task Performance, Visual Attention, and Spatial Learning in a Real-World Navigation Task. Spatial Cognition & Computation.

9. CONSOLE (2024). Correctable Landmark Discovery via Large Models for Vision-Language Navigation.

10. TurnBack (2025). A Comprehensive Benchmark Evaluating LLMs' Geospatial Route Cognition. EMNLP 2025.

11. Schmitt, L., Steinert, S., vd. (2020). A Comparison of Current NLP Libraries for Tokenization, Sentence Splitting and POS-Tagging.

12. NAVCON (2024). A Comprehensive Corpus of Cognitively Motivated Navigation Concepts.

13. VNLP (2024). Turkish NLP Package.

14. Tokens with Meaning (2025). A Hybrid Tokenization Approach for Turkish.

15. Enhancing Turkish Word Segmentation (LORESMT 2024). A Focus on Borrowed Words and Invalid Morphemes.
