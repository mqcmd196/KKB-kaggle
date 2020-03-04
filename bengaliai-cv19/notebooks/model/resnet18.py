class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.resnet18 = models.resnet18(pretrained = True)
        self.head_root = nn.Linear(512, 168) # + softmax
        self.head_vowel = nn.Linear(512, 11) # + softmax
        self.head_consonant = nn.Linear(512, 7) # + softmax
    
    def forward(self, x):
        x = self.resnet18(x)
        head_root = self.head_root(x)
        head_vowel = self.head_vowel(x)
        head_consonant = self.head_consonant(x)
    