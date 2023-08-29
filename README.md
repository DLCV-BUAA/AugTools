# Augtools

## Data Augmentation For Text

### 1 Character

#### 1.1 KeyboardAug

##### 1.1.1 Related Files

- `augtools/extensions/get_keyboard_model_extension.py`
- `augtools/text/char/keyboard_transform.py`

##### 1.1.2 支持的增强操作

替换、交换、删除、插入

##### 1.1.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`。对于被选择的`tokens`，遍历每个`token`，以`aug_char_p`的概率对组成`token`的字符进行增强操作。

对于替换操作，替换的字符为被替换字符在键盘上距离最近的字符中的随机一个。对于插入操作，插入的字符为原位置字符在键盘上距离最近的字符中的随机一个。

这个映射是通过`resource/text/char/keyboard`中的键盘最短距离对应`.json`文件获得的，目前支持键盘类型所属语言体系为德语（de）、英语（en）、西班牙语（es）、法语（fr）、希伯来语（he）、意大利语（it）、荷兰语（nl）、波兰语（pl）、泰语（th）、土耳其语（tr）、俄语（uk）。

对于交换和删除操作，实际上没有用到键盘最短距离的映射，是一个通用操作。交换操作有三种模式可选，分别为adjacent、middle和random。adjacent模式是将选定的字符与其在token中相邻的字符进行交换。middle模式是将选定的字符与和它在同一token中的除了首尾的和它本身之外的某一位置的字符进行交换。random模式是将将选定的字符与和它在同一token中的除了它本身之外的某一位置的字符进行交换。random模式和middle模式相比，多了首尾的两个字符可以作为交换选择。删除操作即为将选定字符删除。



#### 1.2 OcrAug

##### 1.2.1 Related Files

- `augtools/extensions/get_ocr_model_extension.py`
- `augtools/text/char/ocr_transform.py`

##### 1.2.2 支持的增强操作

替换、交换、删除、插入

##### 1.2.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`。对于被选择的`tokens`，遍历每个`token`，以`aug_char_p`的概率对组成`token`的字符进行增强操作。

对于替换操作，替换的字符为被替换字符对应的ocr错误识别字典中的随机一个。对于插入操作，插入的字符为原位置字符对应的ocr错误识别字典中的随机一个。

这个映射是通过`resource/text/char/ocr`中的常见ocr识别错误对应表`.json`文件获得的，目前支持的ocr识别错误仅为英文识别错误。

对于交换和删除操作，实际上没有用到常见ocr识别错误对应表，是一个通用操作。具体实现方式的相关解释参见1.1.3的相关内容。



### 2 Word

#### 2.1 AntonymAug

##### 2.1.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_wordnet_model_extension.py`
- `augtools/text/word/antonym_transform.py`

##### 2.1.2  支持的增强操作

替换

##### 2.1.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`的位置。对于被选择的`tokens`，遍历每一个token，通过调用nltk.corpus模块中的wordnet数据库，获得该token所有可能的反义词，并随机选择其中的一个作为替换单词。



#### 2.2 BackTranslationAug

##### 2.2.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_machine_translation_model_extension.py`
- `augtools/text/word/back_translation_transform.py`

##### 2.2.2 支持的增强操作

替换

##### 2.2.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`的位置。对于被选择的`tokens`，假设原文本属于语言A，遍历每一个token，先调用模型M将token从语言A翻译到语言B得到token1，再调用模型N将token1从语言B翻译回语言A得到token2。需要注意的是，token2很大一部分时候是空的。如果token2是空的，则返回token作为作为增强结果；如果token2非空，则返回token2作为增强结果。



#### 2.3 ContextualWordEmbsAug

##### 2.3.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_fill_mask_model_extension.py`
- `augtools/text/word/context_word_embs_transform.py`

##### 2.3.2 支持的增强操作

插入、替换

##### 2.3.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`的位置。对于被选择的`tokens`，假设原文本属于语言A，遍历每一个token：如果为替换操作，则将该token替换为mask_token；如果为插入操作，则在该token所处位置上插入一个mask_token。处理完后将token列表重新合成文本，丢入fill_mask类别的模型中预测mask_token位置可能对应的单词。mask_token一般为[MASK]或者<mask>，具体是哪一种取决于模型。因为这个增强方法坑比较多，所以下面说说需要注意的地方。

##### 2.3.4 注意事项

为了实现模型的通用性，根据汉语与其他语言的分词以及断句的习惯不同，以及不同模型在对句子进行tokenize操作后生成的形式不同，augmentor需要传入几个特征性的参数。分别是normal,all,prefix和mask_token。实际上大部分英文模型对应的模型类别使用的prefix和mask_token都是固定的，代码中对这些情况进行了考虑。

虽然定义了很多，但是在增强过程中用到的应该只有get_mask_token和get_subword_prefix中的内容。当使用的模型符合标准情况时，normal参数的值为True，代表使用代码中预定义好的prefix和mask_token进行数据处理，normal参数是一个可选参数，默认值就是True。从实验的情况来看，bart、roberta和bert系列的英文语言模型都是符合标准定义的。但是跨语言的LLM以及中文的语言模型的prefix和mask_token和预定义的情况往往不相符。这时需要将normal参数设置为false，并手动传入all、prefix和mask_token参数。

- **all 参数**

all参数是用来告诉augmentor模型对待增强的句子进行tokenize后，是给每个单词都加上了prefix还是只给第一个单词加上了prefix。all参数的作用主要体现在skip_aug函数中。如果tokenizer会给每个词元都加上prefix，那么没有被加上prefix的词元就跳过不作增强处理，如果词元是prefix也不作增强处理。如果tokenizer只给句首的词元加prefix，则不用跳过，直接返回输入的内容即可。具体操作如下：

- **prefix参数**

prefix参数有两个作用。一是是用来告诉augmentor模型在对句子进行tokenize时，给句子中的每个词元添加的首缀的是什么，便于将得到的词元列表进行处理，得到不含prefix的词元列表。

二是在skip_aug的环节判断哪些词元需要被skip掉(一般是在经过tokenize后没有被添加prefix或者词元本身为prefix的时候),不参与增强操作中mask操作对象的筛选。

- **mask_token参数**

mask_token参数的作用实际上就是用模型在预训练时使用的mask符号，通常为<mask>或者[MASK]，在需要替换或者插入的位置占位。不同的模型在预测时承认的mask_token不一样，所以要按照情况传入。

- **tips for finding the value of the three parameters above**

all参数和prefix参数打印分词后的结果查看，mask_token先随便传一个，报错会提示模型要的是哪个，目前为止尝试过的模型的mask_token都是这两个中的一个。

##### 2.3.5 支持的模型及使用时的相关参数（以下为尝试过的一定可以的，没有提及的可能也可以需要自己尝试）

- **英文模型**

  ***\*bert-base-uncased\**** 

   **parameters: language="en"; normal = True**

  ------

  ***\*ccdv/lsg-bart-base-4096\****

   **parameters: language="en"; normal = True**

  ------

  ***\*google/electra-base-generator\****

   **parameters: language="en"; noraml=False ;prefix=""; mask_token=[**MASK**]**

  ------

  ***\*roberta-base\****

   **parameters: language="en"; normal=True**

  ------

  

- **中文模型**

  ***\*uer-chinese-roberta\****

  **parameters: language="cn"; normal=False; prefix=""; mask_token=[**MASK**]**

  ------

  ***\*bert-base-chinese\****

  **parameters:language="cn"; normal=True**

  ------

  

#### 2.4 RandomAug

##### 2.4.1 Related Files

- `augtools/text/word/random_transform.py`

##### 2.4.2 支持的增强操作

交换、替换、删除、裁剪

##### 2.4.3 原理解释

对待增强的句子，以`aug_word_p`的概率随机选出进行增强操作的`tokens`的位置。（实际上就是随机选择aug_word_p*总token数=aug_cnt个token）

对于交换操作，是遍历每个将每个被选中的token，将它与和它原文本中相邻的token进行位置交换。

对于替换操作，是将被选中的token替换为给定字符列表target_words中的一个。默认的target_words列表为["_"]。

对于删除操作，是删除被选中的token。

裁剪操作与交换、替换和删除操作不同，它的操作对象不是按照给定概率选择出的分散的tokens，而是aug_word_p*总token数=aug_cnt个连续的token。裁剪操作就是把这aug_cnt个token裁剪掉。选择aug_cnt个连续token的方法参见`word_transform.py`文件中的`_get_aug_range_idxes`方法，此处不再赘述。



#### 2.5 ReservedAug

##### 2.5.1 Related Files

- `augtools/text/word/reserved_transform.py`

##### 2.5.2 支持的增强操作

替换

##### 2.5.3 原理解释

对待增强的句子，以`aug_word_p`的概率随机选出进行增强操作的`tokens`的位置。对于选择出的token，在传入的参数`reserved_tokens`列表中找到可以用来替换该token的词语们。替换词和被替换词位于同一个列表中，该列表为`reserved_tokens`列表的元素。替换词是在所有的候选中随机选一个。但是当 `generate_all_combinations`参数为True时（只有aug_p=1时，`generate_all_combinations`参数才能设置为True，不然会报错），最后会生成所有可能的替换组合结果。



#### 2.6  SpellingAug

##### 2.6.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_word_spelling_model_extension.py`
- `augtools/text/word/spelling_transform.py`

##### 2.6.2 支持的增强操作

替换

##### 2.6.3 原理解释

对待增强的句子，以`aug_word_p`的概率随机选出进行增强操作的`tokens`的位置。遍历每一个选择出的token，通过错误拼写字典获得所有可能的替换词，随机选择一个作为最终的替换词。错误拼写字典通过加载`augtools/extensions/resource/text/word/spelling`文件夹中的`spelling_en.txt`文件得到。spelling_en.txt文件中只有英文的拼写错误映射，所以该模型目前支持对英文文本的增强操作。



#### 2.7 SplitAug

##### 2.7.1 Related Files

- `augtools/text/word/split_transform.py`

##### 2.7.2 支持的增强操作

分裂

##### 2.7.3 原理解释

对待增强的句子，以`aug_word_p`的概率随机选出进行增强操作的`tokens`的位置。遍历每一个选择出的token，将该token从随机一个位置一分为二分成两个token，用这两个token代替原来的token得到增强结果。



#### 2.8 SynonymAug

##### 2.8.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_wordnet_model_extension.py`
- `augtools/text/word/synonym_transform.py`

##### 2.8.2 支持的增强操作

替换

##### 2.8.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`的位置。对于被选择的`tokens`，遍历每一个token，通过调用nltk.corpus模块中的wordnet数据库，获得该token所有可能的近义词，并随机选择其中的一个作为替换单词。



#### 2.9 TfIdfAug

##### 2.9.1 Related Files

- `augtools/extensions/get_word_tfidf_model_extension.py`
- `augtools/text/word/tfidf_transform.py`

##### 2.9.2 支持的增强操作

插入、替换

##### 2.9.3 原理解释

使用这个增强方法之前，需要在`augtools/extensions/get_word_tfidf_model_extension.py`文件中训练一个tf-idf模型。训练方法是传入一个列表的列表，该列表的元素为一个句子分词后的词元列表，整个列表为一篇文章或者一整个语料的分词汇总结果。将这个列表传入TFIDF模型的train方法中得到每个词元的idf得分和if-idf得分，再调用save方法将结果分别保存下来（默认路径是保存在augtools/extensions/resource目录下）。需要注意的是此处idf的计算和公式给定的有一些出入。可能大部分情况下两者相等，因为大部分情况下一个token只会在一个句子里面出现一次。但是代码中的计算方式实际上和idf的定义是不一样的。训练完毕后，语料中的所有文本在进行增强操作时都可以使用训练得到的idf和tf-idf分数。

与其他word级别的增强操作不同，此处选择进行替换的词元不是均匀分布随机选择的。句子中词元被选中的概率取决于该词元在该句子中的tf-idf分数。（当然各个词元的分数之和一般不等于1，所以将它们作为概率列表使用之前会进行归一化操作）。之所以说是取决于在句子中的tf-idf分数，是因为此处的tf分数的计算公式为：该token在句子中出现的次数/句子中的总token数，并且不同位置出现的token都被认为是不同的token。

在进行替换操作时，被选中的token的替换词会以训练好的tf-idf分数作为概率在所有词元中进行选择top_k个，然后在top_k个中随机选择一个。插入操作也是一样的。这种增强方法的替换完全不考虑每个token的意思，而是以用语料库中最不重要的token替换句子中最不重要的token为操作指导思想。颇具无差别性。



#### 2.10 WordEmbsAug

##### 2.10.1 Related Files

- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/extensions/get_word_emb_glove_extension.py`
- `augtools/extensions/get_word_emb_word2vec_extension.py`
- `augtools/extensions/get_word_emb_fasttext_extension.py`
- `augtools/text/word/word_embs_transform.py`

##### 2.10.2 支持的增强操作

插入、替换

##### 2.10.3 原理解释

对于待增强的句子，以`aug_word_p`的概率选出进行增强操作的`tokens`的位置。

对于插入操作，遍历每一个位置，在所选择的模型的词汇表中随机选一个插入到该位置。

对于替换操作，遍历每一个token，在所选择的模型中挑出与它最接近的且不是它本身的top_k个单词。再在这top_k个单词中随机选择一个作为替换词。

##### 2.10.4 支持的经典嵌入模型

- **glove**
- **word2vec**

- **fasttext**



### 3 Sentence

#### 3.1 AbstSummAug

##### 3.1.1 Related Files

- `augtools/extensions/get_summarization_model_extension.py`
- `augtools/extensions/get_sentence_model_extension.py`
- `augtools/text/sentence/abst_summ_transform.py`

##### 3.1.2 支持的增强操作

替换

##### 3.1.3 原理解释

对输入的文本进行抽象概括，并将抽象概括的结果作为该文本的增强结果。

##### 3.1.4 确定支持的模型

- **英文模型**

  ***\*sshleifer/distilbart-cnn-12-6\****

  ***\*t5-base\****

- **中文模型**

  **\*yihsuan/mt5_chinese_small\***

  **\*dongxq/test_model\***

##### 3.1.5 确定使用之后会报错的模型

**IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese**系列的模型都会报错



#### 3.2 BackTranslationAug

##### 3.2.1 Related Files

- `augtools/extensions/get_machine_translation_model_extension.py`
- `augtools/extensions/get_word_dict_model_extension.py`
- `augtools/text/sentence/back_translation_transform.py`

##### 3.2.2 支持的增强操作

替换

##### 3.2.3 原理解释

将输入的句子S通过模型M从语言A翻译至语言B得到句子S‘，再将S’通过模型N从语言B翻译至语言A得到句子S‘’。将句子S‘’作为句子S的增强结果。

##### 3.2.4 确定支持的模型

- **src_model: 'facebook/wmt19-en-de'**

  **tgt_model: 'facebook/wmt19-de-en'**

- **src_model: 'Helsinki-NLP/opus-mt-en-zh'**

  **tgt_model: 'Helsinki-NLP/opus-mt-zh-en'**

- **src_model: 'Helsinki-NLP/opus-mt-en-zh'**

  **tgt_model: 'Helsinki-NLP/opus-mt-zh-en'**



##### 3.2.5 注意事项

这个增强操作需要两个翻译模型的支持，并且要求句子S的语言和模型M的输入语言、模型N的输出语言一致，模型M的输入语言和模型N的输入语言一致。



#### 3.3 ContextualWordEmbsAug

##### 3.3.1 Related Files

- `augtools/extensions/get_generation_model_extension.py`
- `augtools/extensions/get_sentence_model_extension.py`
- `augtools/text/sentence/abst_summ_transform.py`

##### 3.3.2 支持的增强操作

插入

##### 3.3.3 原理解释

将待增强的文本输入给text_generation类别的模型，生成新的句子。最后将返回的结果作为文本增强结果。

##### 3.3.4 确定支持的模型

- **gpt2**
- **distilgpt2**

##### 3.3.5 确定使用之后会报错的模型

- **mosaicml/mpt-7b-instruct**
- **tiiuae/falcon-7b**



#### 3.4 LambadaAug

##### 3.4.1 Related Files

- `augtools/extensions/get_lambada_model_train_cls.py`
- `augtools/extensions/get_lambada_model_data_processing.py`

- `augtools/extensions/get_lambada_model_run_clm.py`

- `augtools/extensions/get_lambada_model_extension.py`
- `augtools/extensions/get_sentence_model_extension.py`
- `augtools/text/sentence/lambada_transform.py`

##### 3.4.2 支持的增强操作

替换

##### 3.4.3 原理解释

Lambada模型首先用已有数据训练一个分类器。然后对已有数据作格式处理，处理成label[sep]text的形式。将这些数据输入生成器中对生成器进行微调。微调完毕后，用生成器生成指定标签下的指定个数的文本。最后用分类器对生成的文本进行分类，分类结果与标签结果一致的文本就作为该标签下的增强训练数据合并到初始训练数据当中。

Lambada模型在使用前需要运行三个文件，首先运行`get_lambada_model_train_cls.py`训练一个基础的分类器，再运行`get_lambada_model_data_processing.py`将数据处理成label[sep]text的形式。最后运行`augtools/extensions/get_lambada_model_run_clm.py`对生成器进行微调。

使用时需要输入一个标签列表给模型，列表中的元素为需要生成的句子对应的label。

##### 3.3.4 注意事项

运行该增强方式之前需要将resource文件夹改名为resource1。

该方法当微调数据数量小（指三个label，每个label两条数据时）的时候得到的结果并不好，数据量稍微提升一点（指三个label，每个label五条数据），电脑就会直接卡死。（我运行时使用的cpu，电脑配置为16G的运行内存，i9处理器，4060）。鉴于这种情况，这个方法的有效性实际上我无法复现，所以只能把原理和运行方式作一个说明。



#### 3.5 RandomAug

##### 3.5.1 Related Files

- `augtools/extensions/get_shuffle_model_extension.py`
- `augtools/extensions/get_sentence_model_extension.py`
- `augtools/text/sentence/abst_summ_transform.py`

##### 3.5.2 支持的增强操作

替换（交换？）

##### 3.5.3 原理解释

这个Augmentor实际上是词级别RandomAug中的SWAP操作豪华版。它的作用是将句子中指定位置的token根据选中的模式进行交换操作。如果模式是neighbor就随机选择是向左交换还是向右交换。如果模式是left就向左交换，模式是right就向右交换。向左交换则该token与它前面的一个token进行交换，如果该token为第一个token，就和最后一个token进行交换；向右交换则该token与它后面的一个token进行交换，如果该token为最后一个token，就和第一个token进行交换。模式是random就把指定位置的token和句子中随机一个位置（除自己外）的token进行交换。



## Data Augmentation For Image

### 1 Blur 噪声

#### 1.1 Gaussion 高斯噪声

- `augtools/img/transforms/blur/gaussian_blur_transform.py`

#### 1.2 Shot 散粒噪声

- `augtools/img/transforms/blur/shot_noise_transform.py`

#### 1.3 Impulse 脉冲噪声

- `augtools/img/transforms/blur/impulse_noise_transform.py`

#### 1.4 Speckle

- `augtools/img/transforms/blur/speckle_noise_transform.py`

### 2 Filter实现的模糊

此部分属于使用滤波器进行的像素级别的增强。

#### 2.1 Defocus 

- `augtools/img/transforms/blur/defocus_blur_transform.py`

#### 2.2 Glass

- `augtools/img/transforms/blur/glass_blur_transform.py`

#### 2.3 Motion

- `augtools/img/transforms/blur/motion_blur_transform.py`

### 3 特殊天气模拟

模拟特殊天气，如雾、雪等。

#### 3.1 Snow

- `augtools/img/transforms/blur/snow.py`

#### 3.2 Frost

- `augtools/img/transforms/blur/frost.py`

#### 3.3 Fog

- `augtools/img/transforms/blur/fog.py`

### 4 剩余像素级别的调整增强

此部分也属于是像素级别的增强，但无法归类

#### 4.1 Brightness

- `augtools/img/transforms/blur/brightness.py`

#### 4.2 Contrast

- `augtools/img/transforms/blur/contrast.py`

#### 4.3 Saturate

- `augtools/img/transforms/blur/saturate.py`

#### 4.4 Pixelate

- `augtools/img/transforms/blur/pixelate.py`

#### 4.5 Zoom

- `augtools/img/transforms/blur/zoom_blur_transform.py`

### 5 Geometric

此部分属于几何上的增强，如旋转、翻转等，一般带有 BBox 和 Keypoint 等内容的变换。

#### 5.1 ShiftScaleRotate

- `augtools/img/transforms/geometric/ShiftScaleRotate.py`

#### 5.2 Elastic

- `augtools/img/transforms/geometric/ElasticTransform.py`

#### 5.3 Perspective

- `augtools/img/transforms/geometric/Perspective.py`

#### 5.4 PiecewiseAffine

- `augtools/img/transforms/geometric/PiecewiseAffine.py`

#### 5.5 VerticalFlip

- `augtools/img/transforms/geometric/VerticalFlip.py`

#### 5.6 HorizaontalFlip

- `augtools/img/transforms/geometric/HorizontalFlip.py`

#### 5.7 Flip

- `augtools/img/transforms/geometric/Flip.py`

#### 5.8 Transpose

- `augtools/img/transforms/geometric/Transpose.py`

### 6 Crops

#### 6.1 RandomCrop

- `augtools/img/transforms/crops/random_crop_transform.py`

#### 6.2 CenterCrop

- `augtools/img/transforms/crops/center_crop_transform.py`

#### 6.3 Crop

- `augtools/img/transforms/crops/crop_transform.py`

### 4 Synthesis
