def findDecision(obj):
   if obj[0] == 'Sunny':
      if obj[2] == 'High':
         return 'No'
      if obj[2] == 'Normal':
         return 'Yes'
   if obj[0] == 'Rain':
      if obj[3] == 'Weak':
         return 'Yes'
      if obj[3] == 'Strong':
         return 'No'
   if obj[0] == 'Overcast':
      return 'Yes'
