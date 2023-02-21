import json
categories = ['art', 'car', 'computer', 'education', 'family', 'finance', 'food', 'health', 'hobby', 
             'holiday', 'home', 'pet', 'philosophy', 'relationship', 'sport', 'style', 'travel', 'work',
             'youth']
dialogues = []
for category in categories:
    with open(f'{category}_multiwoz_convqa_flan_t5_large_2qa_3sent.json') as f:
        data = json.load(f)
         # The keys are the urls of the task instructions, 
         # such as "https://www.wikihow.com/Adjust-an-Air-Fuel-Mixture-Screw-0:"
        keys = data.keys()
        dialogues.extend([data[key] for key in keys])
        
print(dialogue[0])

# Conversations are seperated with `[SEP]'

# "How to Be Talented in Multiple Areas?[SEP]Setting out to increase your talents and abilities in 
# multiple disciplines is an audacious endeavor.  It’s also very feasible to accomplish.  In fact, 
# it’s far easier to become talented in multiple areas than you may expect.  Practicing the skills
# you want to improve, maintaining a positive mindset, and broadening your base of interest and 
# knowledge can all help you be talented in all sorts of ways.[SEP]What do you need to do to be 
# successful?[SEP]Practice.  Whatever it is you’re trying to be talented at, you know you have to 
# practice.  This is especially true if you hope to be talented in multiple areas[SEP]What do you 
# recommend?[SEP]Practice priority stacking. While you may have multiple interests, use priority 
# stacking to organize your time and resources around the hobbies you are most passionate about.
# Check in with yourself once a week and make sure what you're pursuing is truly what you want 
# and enjoy doing. Don’t worry if you miss a day of practicing one of your talents once in a 
# while[SEP]If you don't like what you're doing, can you try something else?[SEP]Deconstruct the 
# talents you hope to acquire. In order to practice deliberately and efficiently, you need to make 
# sure you are absolutely focused during practice. One way to help maximize the efficiency of your 
# practice time is to deconstruct the talents you hope to improve upon into specific skills[SEP]
# How long will you be able to do that?[SEP]Practice until you can self-correct.  Practice enough 
# so that you are able to both notice and correct errors in your execution of a particular ability.  
# (Once you’ve completed a disciplined practice routine, during which you practice almost daily 
#  for a month, you will likely reach this point.) Moving forward, your practice will become more 
# efficient[SEP]What is the difference between dabbling and practicing?[SEP]Be consistent and 
# persistent. Dabbling and practicing are different things.  Jogging or painting twice a week 
# are fun and healthy things to do, but in order to acquire talent, you need to be more disciplined 
# in your pursuit of improvement[SEP]What is the best way to do that?[SEP]Remove distractions 
# during practice.  Do not rely entirely on willpower to focus adequately during practice time.  
# Here are a few tips to ensure your practice time is free of interruption: Set aside a block 
#   of time devoted exclusively to practice and commit to practicing for that full length of time"
