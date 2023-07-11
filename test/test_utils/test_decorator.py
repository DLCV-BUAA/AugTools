from augtools.utils.decorator import *

def test_lists_process():
    # @lists_process
    # def func(a, b=None):
    #     return a[:1]
    class A:
        @lists_process
        def func(self, a, b=None):
            # print(a)
            return a[:1]
    a = A()
    # c = func([(5, 5), (6, 6)])
    b = a.func([(5, 5), (6, 6)])
    # print(c)
    print(b)

    
    
    
if __name__ == '__main__':
    test_lists_process()