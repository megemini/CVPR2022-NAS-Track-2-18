# CVPR2022 NAS竞赛Track 2第18名方案

方案说明：
1. 本方案基于官方的[baseline](https://aistudio.baidu.com/aistudio/projectdetail/3720834)修改而来。
2. 基于上述方案主要有两个方面的修改：
    * 修改了模型的表达方式
    * 修改了源GPNAS中的kernel function


**模型表达方式**

原始的模型表达方式（编码）太"直观"了，比如"j111231321311311221231121111231000000"这样的一串字符串，而模型的表达远比这来的要复杂，所以我的第一个想法就是如何表达模型。曾经尝试过用embdding的方式(可以参考NLP)，确实能够看到不同的embedding对模型的表示结果不同，但是反映到赛题上结果都很差。

而我这里尝试直接修改原编码字符串，参考代码中的convert_X函数，分数直接由baseline的**0.66提升到0.76**。再简单微调一下就到了**0.78**，ab榜都在18名，而与第一名也就相差1个百分点。

实话讲，不清楚为什么这样做会有这么大的提升，这样刷榜也实在不妥，所以比赛后半端就不再提交了...

这里只能分享一下我修改编码字符串的一些经验：
1. 编码字符串的长短与结果没有必然联系
2. 原始字符串与拓展字符串拼接结果最好
3. 拓展字符串需要能够最大程度区分原始串中"1,2,3"等表示方式，比如用5次方

**GPNAS中的kernel function**

GPNAS源代码中_get_corelation中提供了两个kernel function，而我这里直接使用np.abs，提升在1-2个百分点。
```python
    def _get_corelation(self, mat1, mat2):
        """
        give two typical kernel function
        
        Auto kernel hyperparameters estimation to be updated
        """

        mat_diff = abs(mat1 - mat2)

        if self.c_flag == 1:

            return 0.5 * np.exp(-np.dot(mat_diff, mat_diff) / 16)

        elif self.c_flag == 2:

            # return 1 * np.exp(-np.sqrt(np.dot(mat_diff, mat_diff)) / 12)
            _data = np.dot(mat1, mat2)
            return np.abs(_data)
```

**最后**

这次成绩偶然成分较大，欢迎大家多多交流 ：）
