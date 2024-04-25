# from thefuzz import fuzz
# from thefuzz import process
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# choices = ['T-Shirts Polos', 'Shorts', 'SportSwears', 'Jackets', 'T-Shirts Tops', 'Dresses', 'Skirts', 'Leggings', 'Jerseys', 'Tracksuit', 'Hoodies', 'Pants', 'Tights']
# sentence = "i want it to be red and sleeveless with adidas logo on it"
# res = []
# stop_words = set(stopwords.words('english'))
# for word in nltk.word_tokenize(sentence):
#     if word not in stop_words and \
#     word.isalnum() and \
#     process.extractOne(word, choices)[1] > 80:
#         print(word)
#         res.append(process.extractOne(word, choices))
# print(res)

# original = ['T-Shirts & Polos', 'Shorts', 'SportSwears', 'Jackets', 'T-Shirts and Tops', 'Dresses', 'Skirts', 'Leggings', 'Jerseys', 'Tracksuit', 'Hoodies', 'Pants', 'Tights']
# modified = ['tshirt_and_polo', 'short', 'sportswear', 'jacket', 'tshirts_and_top', 'dress', 'skirt', 'legging', 'jersey', 'tracksuit', 'hoodie', 'pant', 'tight']
# dict = {}
# for i in range(len(original)):
#     dict[original[i]] = modified[i]
# print(dict)

# d = ['tshirt_and_polo', 'short', 'sportswear', 'jacket', 'tshirt_and_top', 'dress', 'skirt', 'legging', 'jersey', 'tracksuit', 'hoodie', 'pant', 'tight']
# s = """
# ##########################################
# ##########################################
# ##########################################
# global fashionIQ_val_{cate}_index_features
# fashionIQ_val_{cate}_index_features = torch.load(
#     data_path / 'fashionIQ_val_{cate}_index_features.pt', map_location=device).type(data_type).cpu()

# global fashionIQ_val_{cate}_index_names
# with open(data_path / 'fashionIQ_val_{cate}_index_names.pkl', 'rb') as f:
#     fashionIQ_val_{cate}_index_names = pickle.load(f)

# # global fashionIQ_test_{cate}_index_features
# # fashionIQ_test_{cate}_index_features = torch.load(
# #     data_path / 'fashionIQ_test_{cate}_index_features.pt', map_location=device).type(data_type).cpu()

# # global fashionIQ_test_{cate}_index_names
# # with open(data_path / 'fashionIQ_test_{cate}_index_names.pkl', 'rb') as f:
# #     fashionIQ_test_{cate}_index_names = pickle.load(f)

# global fashionIQ_{cate}_index_features
# global fashionIQ_{cate}_index_names
# # fashionIQ_{cate}_index_features = torch.vstack(
#     # (fashionIQ_val_{cate}_index_features, fashionIQ_test_{cate}_index_features))
# fashionIQ_{cate}_index_features = fashionIQ_val_{cate}_index_features
# # fashionIQ_{cate}_index_names = fashionIQ_val_{cate}_index_names + fashionIQ_test_{cate}_index_names
# fashionIQ_{cate}_index_names = fashionIQ_val_{cate}_index_names
# """
# s = """
#     elif dress_type == "{cate}":
#         if target_name == "":
#             index_names = fashionIQ_{cate}_index_names
#             index_features = fashionIQ_{cate}_index_features
#         else:
#             index_names = fashionIQ_val_{cate}_index_names
#             index_features = fashionIQ_val_{cate}_index_features
# """
# for i in d:
#     print(s.format(cate=i))
