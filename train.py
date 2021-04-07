import torch
import pickle

from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from time import time

from model import Encoder, RNNAttnDecoder

from collections import Counter

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.totalImages, self.totalLabels = [], []

        with Path(cfg['label_path']).open('r', encoding='utf8') as f:
            for line in f:
                fn, label = line.strip().split('\t')
                self.totalLabels.append(label)
                self.totalImages.append(Image.open(Path(cfg['image_path'])/fn))

        vocab = {'token2id': {'[START]':0, '[FINISH]':1},
                 'id2token': {0:'[START]', 1:'[FINISH]'}}
        cnt = 2  # start token : 0 / finish token : 1
        for label in self.totalLabels:
            for l in label:
                if l not in vocab['token2id']:
                    vocab['id2token'][cnt] = l
                    vocab['token2id'][l] = cnt
                    cnt += 1 
        self.vocab_size = len(vocab['token2id'])
        
        with open('{}/captcha_vocab.dict'.format(cfg['save_path']), 'wb') as f:
            pickle.dump(vocab, f)
        
        transform = transforms.Compose([transforms.Resize((50, 110)),
                                        transforms.ToTensor()])

        self.totalImages = [transform(img) for img in self.totalImages]
        temp = []
        for s in self.totalLabels:
            #  # start token : 0 / finish token : 1
            temp.append(torch.LongTensor([0]+[vocab['token2id'][token] for token in s]+[1]))
        self.totalLabels = torch.stack(temp)

    def __len__(self):
        return len(self.totalImages)

    def __getitem__(self, idx):
        return self.totalImages[idx], self.totalLabels[idx]

class CaptchaRecog(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = CaptchaDataset(cfg)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                    num_workers=4, 
                                                    shuffle=True,
                                                    batch_size=cfg['bs'],
                                                    drop_last=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder(num_rnn_layers=1, rnn_hidden_size=64, dropout=0.)

        self.decoder = RNNAttnDecoder(input_vocab_size=self.dataset.vocab_size,
                                      hidden_size=64,
                                      output_size=self.dataset.vocab_size,
                                      num_rnn_layers=1, dropout=0.)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        encoderParams = list(filter(lambda p:p.requires_grad, self.encoder.parameters()))
        decoderParams = list(filter(lambda p:p.requires_grad, self.decoder.parameters()))
        self.encoderOptimizer = torch.optim.Adam(encoderParams, lr=cfg['lr'])
        self.decoderOptimizer = torch.optim.Adam(decoderParams, lr=cfg['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

        self.transform = transforms.Compose([   
                                            transforms.Resize((50, 110)),
                                            transforms.ToTensor()
                                            ])

    def train(self):
        self.encoder.train()
        self.decoder.train()

        clip=10.
        min, best = 999, None
        for e in range(self.cfg['epochs']):
            acc = 0
            pbar = tqdm(self.dataloader, total=len(self.dataset)//self.cfg['bs'])
            for i, data in enumerate(pbar, 1):
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)
                loss = 0

                self.encoderOptimizer.zero_grad()
                self.decoderOptimizer.zero_grad()

                maxLen = labels.size()[1]

                init_hidden = self.encoder.initHidden(self.cfg['bs'], use_cuda=self.device.type=='cuda')
                encoder_outputs,encoder_hidden = self.encoder(images, init_hidden)

                last_hidden = encoder_hidden
                last_ht = torch.zeros(self.cfg['bs'], self.decoder.hidden_size).to(self.device)
                outputs = torch.zeros((self.cfg['bs'], maxLen-1)).long().to(self.device)

                for di in range(maxLen-1):
                    inp = labels[:,di]
                    target = labels[:,di+1]
                    output, last_ht, last_hidden, alpha = self.decoder(inp,
                                                                    last_ht,
                                                                    last_hidden,
                                                                    encoder_outputs)
                    outputs[:,di] = output.max(1)[1].data
                    loss += self.criterion(output, target)

                acc += sum(torch.all(torch.eq(outputs, labels[:,1:]), dim=1)).item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
                self.encoderOptimizer.step()
                self.decoderOptimizer.step()
                
                if loss < min:
                    best = {'encoder':self.encoder.state_dict(),
                            'decoder':self.decoder.state_dict()}

                pbar.set_description("epoch[{:3d}] iter[{:4d}] loss[{:.4f}] acc[{:.3f}]".format(
                                            e, i, loss, acc/i/self.cfg['bs']))
        
        torch.save(best, '{}/best.pt'.format(self.cfg['save_path']))

    def pred(self, img, gif=False):
        self.vocab = pickle.load(open('{}/captcha_vocab.dict'.format(self.cfg['save_path']),'rb'), 
                                 encoding='utf8')
        self.encoder.eval()
        self.decoder.eval()
        t = time()
        if isinstance(img, str):
            img = Image.open(img)
        bag = []
        if gif:
            for i in range(img.n_frames):
                img.seek(i)
                im = img.copy().convert("RGB")
                bag.append(im)
            img = torch.stack([self.transform(im) for im in bag])
        else:
            img = self.transform(img).unsqueeze(0)
        inputs = img.to(self.device)

        bs = inputs.size(0)

        init_hidden = self.encoder.initHidden(bs, use_cuda=self.device.type=='cuda')
        encoder_outputs,encoder_hidden = self.encoder(inputs,init_hidden)

        last_hidden = encoder_hidden
        last_ht = torch.zeros(bs, self.decoder.hidden_size).to(self.device)
        outputs = torch.zeros((bs, 5)).long().to(self.device)
        
        inp = torch.zeros((bs,)).long().to(self.device)
        for di in range(5):
            output, last_ht, last_hidden, _ = self.decoder(inp, last_ht, last_hidden, encoder_outputs)
            inp = output.max(1)[1]
            outputs[:,di] = inp.data

        print('CAPTCHA solved in {:.4f}sec'.format(time()-t))
        ret = []
        for i in range(bs):
            s = ''.join([self.vocab['id2token'][id] for id in outputs.tolist()[i]])
            ret.append(s)
        c = Counter(ret)
        return c.most_common()[0][0]


if __name__ == "__main__":
    cfg = {
        'bs': 4,
        'lr': 0.002,
        'epochs' : 50,
        'save_path': 'result',
        'image_path' : 'data/train_data',
        'label_path' : 'data/train_data.txt',
    }
    cr = CaptchaRecog(cfg)
    cr.train()
    
    from pathlib import Path
    for x in Path('data/test_data').glob('*.jpg'):
        print(x, ':', cr.pred(str(x)))