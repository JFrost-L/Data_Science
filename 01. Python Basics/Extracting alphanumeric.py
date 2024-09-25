'''
주어진 문장에서 alphanumeric 문자를 추출하여,
list를 만들어서 출력하라.
• 해당 리스트의 각 문자에 대하여 count 를
저장하는 dict를 만들어서 출력하라.
• 해당 dict를 다시 문자의 count 순으로 소팅하고
출력하라.
'''
def get_char_dict(sentence):
    char_list = [c for c in sentence if c.isalnum()]
    print(char_list)
    print()
    
    char_count_dic = {c:char_list.count(c) for c in char_list }
    print(char_count_dic)
    
    
    sorted_dic = dict(sorted(char_count_dic.items(), key=lambda x: x[1], reverse=True))
    print(sorted_dic)
    
if __name__ == '__main__':
    sentence = "주어진 문장에서 alphanumeric 문자를 추출하여,\
    list를 만들어서 출력하라.\
    • 해당 리스트의 각 문자에 대하여 count 를\
    저장하는 dict를 만들어서 출력하라.\
    • 해당 dict를 다시 문자의 count 순으로 소팅하고\
    출력하라."
    
    
    get_char_dict(sentence)
    '''
    for str in list:
        dic[str]=0
        
    for str in list:
        if str in dic:
            dic[str]+=1
    print(dic)
    
    print(sorted_dic)
    '''          