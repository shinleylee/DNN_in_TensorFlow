# # list
# exampleList = [1,2,3,4]
# print(exampleList[0])
# exampleList.append('haha')
# print(exampleList)
#
# # dictionary
# exampleDictionary = {"key0":"value0", "key1":"value1"}
# print(exampleDictionary['key1'])
# exampleDictionary['newKey'] = 'newValue'
# print(exampleDictionary)


# class
class Vechicle:
    def __init__(self, number_of_wheels, max_velocity):
        self.number_of_wheels = number_of_wheels
        self.max_velocity = max_velocity

    @property
    def number_of_wheels(self):
        return self._number_of_wheels

    @number_of_wheels.setter
    def number_of_wheels(self, new_number_of_wheels):
        self._number_of_wheels = new_number_of_wheels


class sonofClassVechicle(Vechicle):
    def __init__(self,number_of_wheels, max_velocity):
        Vechicle.__init__(self, number_of_wheels, max_velocity)

car = Vechicle(4,120)
print(car.number_of_wheels)
car.number_of_wheels = 6
print(car.number_of_wheels)

carSon = sonofClassVechicle(2,60)
print(carSon.number_of_wheels)
