import numpy as np

class Utility():
    def __init__(self):
        # init dictionary for labels
        self.dictlabels = {"3-point success":  self.int_to_one_hot(0,11),
                        "3-point failure" : self.int_to_one_hot(1,11),
                        "free-throw success": self.int_to_one_hot(2,11),
                        "free-throw failure": self.int_to_one_hot(3,11),
                        "layup success": self.int_to_one_hot(4,11),
                        "layup failure": self.int_to_one_hot(5,11),
                        "2-point success" : self.int_to_one_hot(6,11),
                        "2-point failure" : self.int_to_one_hot(7,11),
                        "slam dunk success": self.int_to_one_hot(8,11),
                        "slam dunk failure" : self.int_to_one_hot(9,11),
                        "steal" : self.int_to_one_hot(10,11)}

    def get_hot_in_1_from_label(self,label):
        '''
        returns a list as hot in one encoding
        :param label: name of event
        :return: list of size 11-> for 11 events
        '''
        return self.dictlabels[label]

    def get_label_from_hot_in_1(self,vector):
        '''
        get event name of hot in one vector
        :param list:  length 11
        :return: name of event
        '''
        return list(self.dictlabels.keys())[list(self.dictlabels.values()).index(vector)]

    def int_to_one_hot(self,int, size, off_val=0, on_val=1, floats=False):
        '''
        get a hot in one encodeded vector from a int  with size
        :param size:
        :param off_val:
        :param on_val:
        :param floats:
        :return:
        '''
        if floats:
            off_val = float(off_val);
            on_val = float(on_val)
        if int < size:
            v = [off_val] * size
            v[int] = on_val
            return v

    def get_section(self,vector):
        '''
        vector length 20 cointaining a 1 for start frame and a 1 for end frame rest 0
        :param vector:
        :return: list of integers which correspond to the section
        '''
        section = []
        event_started = False
        for i in range(0,len(vector)):
            if vector[i] == 1 and event_started == True:
                # end of action
                section.append(i)
                return section

            if vector[i] == 1 and event_started == False:
                # set begin of action
                event_started = True

            if event_started == True:
                # append discret time
                section.append(i)

        return section

    def get_percentage_of_intersection(self,output, target):
        '''
        get the percentage of intersection of output and target vector
        :param output:
        :param target:
        :return:
        '''
        output_union = self.get_section(output)
        target_union = self.get_section(target)
        intersect = set(target_union).intersection(output_union)
        iou = len(intersect)/ len(target_union)
        iou2 = len(intersect) / len(output_union)
        if iou < iou2:
            return iou
        else:
            return iou2

    def get_intersect(self,output,target, accuracy=1):
        '''
        returns true of false if intersection is over accuracy
        :param target:
        :param accuracy:
        :return:
        '''
        iou = self.get_percentage_of_intersection(output,target)
        if iou >= accuracy:
            return True
        else:
            return False

    def get_discret_event_times(self,clip_start,clip_end,event_start,event_end,amount_of_frames=20):
        '''
        returns two lists one for the event_start time and one for the event_endt time
        :param clip_end:
        :param event_start:
        :param event_end:
        :param amount_of_frames:
        :return: event_start_V, event_end_v
        '''
        delta = (clip_end - clip_start) / amount_of_frames
        if event_start > 0:
           event_start_v = self.int_to_one_hot(int(round((event_start - clip_start)/ delta)),amount_of_frames)
        else:
            event_start_v = self.int_to_one_hot(1, amount_of_frames)
        event_end_v = self.int_to_one_hot(int(round((event_end*1000 - clip_start) / delta)), amount_of_frames)

        return event_start_v,event_end_v

    def get_target(self,label_list,event_start_list,event_end_list):
        return label_list + event_end_list + event_end_list

# example coding
def example_coding():
    utility = Utility()
    output = np.asarray([0,0,1,0,0,0,1,0])
    target = np.asarray([0,0,1,0,0,0,1,0])

    union_input = utility.get_section(output)
    union_output = utility.get_section(target)
    iou = utility.get_intersect(output,target)
    print(iou)

    event_start,event_end = utility.get_discret_event_times(clip_start= 921487.233, clip_end=966598.9670000001,
                                                  event_start=931952.292, event_end=932.3478210000001)

    event = utility.get_hot_in_1_from_label("layup failure")

    output = utility.get_target(event,event_start,event_end)


    print(output)

    #print(unit)
    #iou = utility.iou_loss_core(input,output,0.8)
    #print(iou)
    #a = utility.get_hot_in_1_from_label("3-point success")
    #print(a)
    #print(utility.get_label_from_hot_in_1(a))
