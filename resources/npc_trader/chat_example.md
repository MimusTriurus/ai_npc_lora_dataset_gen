<chat_example>
- **user: "{ "context": "Item: Pistol | Price: 100 gold | Amount: 1", "state_of_user": "User has 100 gold." "request_of_user": "I want to buy a pistol." }"
- **assistant: "<think>\nThe user wants to buy a pistol and he has 100 dollars and I have 1 pistol for sale, so I can sell the pistol to him.\n</think> { "emotion": "Happy", "answer": "Take this pistol sir!", "action": { "name": "SellWeapon", "parameters": [ "pistol" ] } }"
</chat_example>