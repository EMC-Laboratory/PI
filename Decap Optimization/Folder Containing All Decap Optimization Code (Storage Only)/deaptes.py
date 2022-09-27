from deap import creator
import PopInit
import config2

opt = config2.Config()
creator.create("Foo",list, bar=dict, spam=1)
x = Foo(list)
print(isinstance(x,list))