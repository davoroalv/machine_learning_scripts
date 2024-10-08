class Menu:
  def __init__(self, name, items, start_time, end_time):
    self.name = name
    self.items = items
    self.start_time = start_time
    self.end_time = end_time

  def __repr__(self):
    return f"The {self.name} menu is served from {self.start_time} to {self.end_time}."

  def calculate_bill(self, purchased_items):
    total = 0
    for i in purchased_items:
      total+=self.items[i]
    print("The total is : $"+str(total))  
    return total

brunch = Menu('brunch', {
  'pancakes': 7.50, 'waffles': 9.00, 'burger': 11.00, 'home fries': 4.50, 'coffee': 1.50, 'espresso': 3.00, 'tea': 1.00, 'mimosa': 10.50, 'orange juice': 3.50
  }, 11,16)

early_bird = Menu('early_bird', {
  'salumeria plate': 8.00, 'salad and breadsticks (serves 2, no refills)': 14.00, 'pizza with quattro formaggi': 9.00, 'duck ragu': 17.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 1.50, 'espresso': 3.00,
  }, 15, 18)

dinner = Menu('dinner', {
  'crostini with eggplant caponata': 13.00, 'caesar salad': 16.00, 'pizza with quattro formaggi': 11.00, 'duck ragu': 19.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 2.00, 'espresso': 3.00,
  }, 17, 23)

kids = Menu('kids', {
  'chicken nuggets': 6.50, 'fusilli with wild mushrooms': 12.00, 'apple juice': 3.00
  }, 11, 21)

# print(brunch)
# print(dinner)
# print(kids)

# brunch.calculate_bill(['pancakes','home fries','coffee'])

# early_bird.calculate_bill(['salumeria plate','mushroom ravioli (vegan)'])

class Franchise:
  def __init__(self, address, menus):
    self.address = address
    self.menus = menus
  def __repr__(self):
    return self.address

  def available_menus(self,time):
    menus_avail = []
    for i in self.menus:
      # print(i.name,i.start_time, i.end_time)
      if time >= i.start_time and time <= i.end_time:
        # print(i.name,i.start_time, i.end_time)
        menus_avail.append(i)
    # print(menus_avail)
    return menus_avail

flagship_store = Franchise("1232 West End Road", [brunch, early_bird, dinner, kids])
new_installment = Franchise("12 East Mulberry Street", [brunch, early_bird, dinner, kids])

# print(flagship_store.menus)

# print(flagship_store.available_menus(12))
# print(flagship_store.available_menus(17))

class Business:
  def __init__(self,name, franchises):
    self.name = name
    self.franchises = franchises
  
arepas_menu = Menu("arepas_menu", {
  'arepa pabellon': 7.00, 'pernil arepa': 8.50, 'guayanes arepa': 8.00, 'jamon arepa': 7.50
  }, 10,20)

arepas_place = Franchise("189 Fitzgerald Avenue",[arepas_menu])

first_biz = Business("Take a' Arepa", arepas_place)
