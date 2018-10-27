class Utility():
    def __init__(self):
        # init dictionary for labels
        self.dictlabels = {"3-point success":  Utility.int_to_one_hot(0,11),
                        "3-point failure" : Utility.int_to_one_hot(1,11),
                        "Free-throw success": Utility.int_to_one_hot(2,11),
                        "Free-throw failure": Utility.int_to_one_hot(3,11),
                        "Layup success": Utility.int_to_one_hot(4,11),
                        "Layup failure": Utility.int_to_one_hot(5,11),
                        "2-point success" : Utility.int_to_one_hot(6,11),
                        "2-point failure" : Utility.int_to_one_hot(7,11),
                        "Slam dunk success": Utility.int_to_one_hot(8,11),
                        "Slam dunk failure" : Utility.int_to_one_hot(9,11),
                        "Steal" : Utility.int_to_one_hot(10,11)}

    def get_hot_in_1_from_label(self,label):
        '''
        returns vector as hot in one encoding
        :param label: name of event
        :return: vector of size 11-> for 11 events
        '''
        return self.dictlabels[label]

    def get_label_from_hot_in_1(self,vector):
        '''
        get event name of hot in one vector
        :param vector:  vector length 11
        :return: name of event
        '''
        return list(self.dictlabels.keys())[list(self.dictlabels.values()).index(vector)]

    def int_to_one_hot(int, size, off_val=0, on_val=1, floats=False):
        if floats:
            off_val = float(off_val);
            on_val = float(on_val)
        if int < size:
            v = [off_val] * size
            v[int] = on_val
            return v

# example coding
#utility = Utility()
#a = utility.get_hot_in_1_from_label("3-point success")
#print(a)
#print(utility.get_label_from_hot_in_1(a))
