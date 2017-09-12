import random

class Zoo:
    name = 'Moscow Zoo'
    @classmethod
    def info(cls):
        return cls.name

class Predator(Zoo):
    kind = 'predator'
    @classmethod
    def get_kind(cls):
        return cls.kind
    def info(self):
        return 'I am a %s in a %s' % (self._kind, self.name)

class Herb(Zoo):
    kind = 'herbivore'
    @classmethod
    def get_kind(cls):
        return cls.kind
    def info(self):
        return 'I am a %s in a %s' % (self._kind, self.name)

class Tiger(Predator):
    def __init__(self, spec):
        self._spec = spec
        self.likely = random.uniform(0.7, 1.0)
    def where(self):
        return 'A %s is upper in a food chain whith likelyhood %.1f' % (self._spec, self.likely)
    def info(self):
        return 'I am a %s and a %s in a %s' % (self._spec, self.kind, self.name)

class Owl(Predator):
    def __init__(self, spec):
        self._spec = spec
        self.likely = random.uniform(0.6, 1.0)
    def where(self):
        return 'A %s is upper in a food chain whith likelyhood %.1f' % (self._spec, self.likely)
    def info(self):
        return 'I am a %s and a %s in a %s' % (self._spec, self.kind, self.name)

class Gazelle(Herb):
    def __init__(self, spec):
        self._spec = spec
        self.likely = random.uniform(0.0, 0.5)
    def where(self):
        return 'A %s is lower in a food chain whith likelyhood %.1f' % (self._spec, self.likely)
    def info(self):
        return 'I am a %s and a %s in a %s' % (self._spec, self.kind, self.name)

class Rabbit(Herb):
    def __init__(self, spec):
        self._spec = spec
        self.likely = random.uniform(0.0, 0.4)
    def where(self):
        return 'A %s is lower in a food chain whith likelyhood %.1f' % (self._spec, self.likely)
    def info(self):
        return 'I am a %s and a %s in a %s' % (self._spec, self.kind, self.name)

class Chain:
    def __init__(self, part):
        self.part = part
    #Вероятность того, в какой части пищевой цепи находтся животное,
    #зависит от того, хищник это или травоядное, а также от вида животного
    def place(self):
        return self.part.where()


tig = Tiger('tiger')
ow = Owl('owl')
gaz = Gazelle('gazelle')
rab = Rabbit('rabbit')
print(tig.info())
print(ow.info())
print(gaz.info())
print(rab.info())
ch1 = Chain(tig)
print(ch1.place())
ch2 = Chain(ow)
print(ch2.place())
ch3 = Chain(gaz)
print(ch3.place())
ch4 = Chain(rab)
print(ch4.place())
