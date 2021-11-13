from personal_data.finder import PersonalDataFinder

finder = PersonalDataFinder()

text = "Hi John, i'm Michael Wick. My ipv4 address is 192.168.5.18 and " \
       "ipv6 is 2001:0000:4136:e378:8000:63bf:3fff:fdd2." \
       " Reply to me on michael.petrolini@omigrade.it or michael.petrolini@studenti.unipr.it" \
       " o trovarmi in 4 Saffron Hill Road"

print(finder.look_for_emails(text))
print(finder.look_for_ip_address(text))
print(finder.look_for_address(text))
print(finder.look_for_names(text))
print(finder.look_for_surnames(text))
