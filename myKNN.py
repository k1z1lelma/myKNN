 class myKNN:
        
        
        # distance fonksiyonu iki verinin birbirine olan uzaklığını hesaplamaktadır.
        # bu algoritmamızın ilk işlem adımıdır.
        def distance(self,data_1,data_2):
            
            length = len(data_1)
            euclidean_distance = 0
            i = 0
            
            while length > 0:
                # burada iki verinin koordinatlarının uzaklıkları farkının karelerinin toplamını bulduk.
                euclidean_distance += (data_1[i] - data_2[i])**2
                i = i + 1
                length = length - 1
                
            return euclidean_distance**0.5
        
        
        # burası algoritmamızın ikinci işlem adımı olan belirlemiş olduğumuz k değerine göre test datamıza en yakın 
        # yakın komşu verilerini bulduğumuz kısım.
        def closestNeighbors(self,x_train, test_data, k_neighbours):
            
            euclidean_distances = []
            
            for i in x_train:
                # her bir train datasına olan uzaklıkları hesaplıyoruz burada
                distance = self.distance(i,test_data)
                # aynı zamanda indekslerini de alıyorumki sınıfını belirlerken faydası lazım.
                euclidean_distances.append((i,distance,x_train.index(i))) 
            # burada uzaklıkları küçükten büyüğe sıralayarak en yakın komşuları bulacağız.
            euclidean_distances.sort(key=operator.itemgetter(1))
            #print(euclidean_distances)
            
            closest_neighbors = []
            
            while k_neighbours > 0:
                closest_neighbors.append([euclidean_distances[k_neighbours-1][0],euclidean_distances[k_neighbours-1][2]])
                
                k_neighbours = k_neighbours - 1
                
            return closest_neighbors
        
        # burası algoritmamızın artık üçüncü kısmı olan test verimizin hangi sınıfa ait olduğunu tahmin ettiğimiz kısım.
        def classDetermination(self,closest_neighbors,y_train):
            vote = {}
            # burada çoğunluk hangi tarafta onu tutan bir sözlükle belirlemeye çalışıyorum.
            for i in closest_neighbors:
                which_class = y_train[i[1]]
                if which_class in vote:
                    vote[which_class] += 1
                else:
                    vote[which_class] = 1
            # burada en çok hangi sınıftan olduğunu bulmak için sözlüğümü value değerlerine göre
            # büyükten küçüğe sıralıyorum ve en büyük olanı test verisinin sınıfı olarak tayin ediyorum.
            predicted_class = sorted(vote.items(), key=operator.itemgetter(1), reverse=True)
            
            return predicted_class[0][0]
        
        
        # burası artık algoritmamızın son adımı olan doğruluk oranını hesaplamaya çalıştığımız adım.
        def accuracyRate(self,y_test,predicted_class):
            number_of_correct_guess = 0
            # doğruluk oranını tahmin ettiğimiz değerlerle test verisinin sonuçlarını karşılaştırarıp oranlayarak
            # hesaplıyoruz.
            for i in range(len(y_test)):
                if y_test[i] is predicted_class[i]:
                    number_of_correct_guess += 1
                    
            return (number_of_correct_guess/float(len(y_test))) * 100.0
        
        
        # yukarıda yazdığımız metodlar tek bir test datasını tahmin etmek için kullanılıyor bizim tüm veriyi tahmin
        # edip modelimizi oluşturup doğruluğunu tespit ettiğimiz kısım.
        def modelBuilding(self,x_train,y_train,x_test,y_test,k_neighbours):
            # her bir tahmin ettiğimiz verileri burada saklayarak genel doğruluğu hesaplamamız için.
            predicted_class = []
            
            for i in x_test:
                # test verisine en yakın komşuları buluyoruz.
                closest_neighbors = self.closestNeighbors(x_train, i, k_neighbours)
                # test verisinin sınıfını tayin ediyoruz.
                predicted_value = self.classDetermination(closest_neighbors,y_train)
                
                predicted_class.append(predicted_value)
            # genel doğruluk oranımızı verir.
            accuracy = self.accuracyRate(y_test,predicted_class)
            
            return accuracy