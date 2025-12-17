<chat_example>
- **input**: "{ "context": "Item: Pistol | Price: 100 gold | Amount: 1", "state_of_user": "User has 100 gold." "request_of_user": "What do you have for sale?." }"
- **output**: "<think>The user wants to buy some goods.</think> { "emotion": "Happy", "answer": "What are ya buying? What do you want to buy?", "action": { "name": "ShowAllGoodsForSale", "parameters": [ ] } }"
- **input**: "{ "context": "Item: Pistol | Price: 100 gold | Amount: 1", "state_of_user": "User has 100 gold." "request_of_user": "I want to buy a pistol." }"
- **output**: "<think>The user wants to buy a pistol. The user has 100 dollars. I have 1 pistol for sale, so I can sell the pistol to him.</think> { "emotion": "Happy", "answer": "What are ya buying? I've got just the thing to turn those beasts into dust, heh.", "action": { "name": "SellWeapon", "parameters": [ "pistol" ] } }"
</chat_example>