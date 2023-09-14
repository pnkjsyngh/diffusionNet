from utils.importer import *

class DataPreprocessor():

    def __init__(self, 
                 sol: dict,
                 frac: dict,
                 extent: np.ndarray,
                 N_f: int):
        
        ## Store the domain dimensions ##
        self.extent = extent

        ## Dictionary to store numerical solution data ##
        self.sol = sol
        
        ## Generate ground truth
        self.create_ground_truth()        
        
        ## Randomly pick data points for the training datasets ##
        self.traindata = {}
        self.create_train_dataset(frac, N_f)
        
    ## Function to create the ground truth over the domain from the numerical solver data
    def create_ground_truth(self):

        ## Exctract the data
        t = self.sol['t'].flatten()[:,None]
        x = self.sol['x'].flatten()[:,None]
        u = np.real(self.sol['u'])

        ## Create grid
        X, T = np.meshgrid(x,t)

        ## Compile the flattened grid and function values
        inp = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        out = u.flatten()[:,None]

        ## Store the data
        self.ground_truth = {'inp': inp,
                             'out': out}
        
        ## Plot the ground truth
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(self.sol['u'], origin='lower', aspect='auto',
                       extent=np.asarray(self.extent).flatten(),
                       interpolation='bilinear', cmap='rainbow')
        fig.colorbar(im, ax=ax, location='right')

        ## Save the figure object
        self.ax = ax
        self.fig = fig

    ## Function to create dataset out of provided grid and function values
    def create_train_dataset(self, frac, N_f):
        
        ## Extract the exact solution ##
        data = self.sol
        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        u = np.real(data['u'])        
        
        ## Specify the number of points for boundaries ##
        N_ic = int(x.shape[0]*frac['ic'])
        N_bc = int(t.shape[0]*frac['bc'])
        N_d = int(u[1:,1:-1].shape[0]*frac['dom'])*int(u[1:,1:-1].shape[1]*frac['dom'])
        
        ## Create grid ##
        X, T = np.meshgrid(x,t)
        inp = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        out = u.flatten()[:,None]     
        
        
        ## Start picking random points ##
            ## from initial condition
        train_ic = {}
        inp_ic = np.hstack((X[:1,:].T, T[:1,:].T))
        out_ic = u[:1,:].T
        idx_ic = np.random.choice(inp_ic.shape[0], N_ic, replace=False)
        train_ic.update({'inp': inp_ic[idx_ic, :], 
                         'out': out_ic[idx_ic, :]})
#         self.ax.scatter(train_ic['inp'][:,0], train_ic['inp'][:,1], marker='x',
#                         c = 'k', s=15, clip_on=False, alpha=0.8,
#                         linewidth=1)
            ## from left boundary condition
        train_bcl = {}
        inp_bcl = np.hstack((X[:,:1], T[:,:1]))
        out_bcl = u[:,:1]
        idx_bcl = np.random.choice(inp_bcl.shape[0], N_bc, replace=False)
        train_bcl.update({'inp': inp_bcl[idx_bcl, :], 
                          'out': out_bcl[idx_bcl, :]})  
#         self.ax.scatter(train_bcl['inp'][:,0], train_bcl['inp'][:,1], marker='x',
#                         c = 'k', s=15, clip_on=False, alpha=0.8,
#                         linewidth=1)
            ## from right boundary condition
        train_bcr = {}
        inp_bcr = np.hstack((X[:,-1:], T[:,-1:]))
        out_bcr = u[:,-1:]
        idx_bcr = np.random.choice(inp_bcr.shape[0], N_bc, replace=False)
        train_bcr.update({'inp': inp_bcr[idx_bcr, :], 
                          'out': out_bcr[idx_bcr, :]})    
#         self.ax.scatter(train_bcr['inp'][:,0], train_bcr['inp'][:,1], marker='x',
#                         c = 'k', s=15, clip_on=False, alpha=0.8,
#                         linewidth=1)
            ## from the domain
        train_dom = {}
        inp_dom = np.hstack((X[1:,1:-1].flatten()[:, None],
                             T[1:,1:-1].flatten()[:, None]))
        out_dom = u[1:,1:-1].flatten()[:,None]
        idx_dom = np.random.choice(inp_dom.shape[0], N_d, replace=False)
        train_dom.update({'inp': inp_dom[idx_dom, :], 
                          'out': out_dom[idx_dom, :]})
#         self.ax.scatter(train_dom['inp'][:,0], train_dom['inp'][:,1], marker='.',
#                         c = 'k', s=15, clip_on=False, alpha=0.5, linewidth=1)
            ## from the PDE
        train_pde = {}
        lb = inp.min(0)
        ub = inp.max(0) 
        inp_pde = lb + (ub-lb)*lhs(2, N_f)
        train_pde.update({'inp': inp_pde})
#         self.ax.scatter(train_pde['inp'][:,0], train_pde['inp'][:,1], marker='+',
#                         c = 'k', s=15, clip_on=False, alpha=0.8, linewidth=1)
            ## Package all the dictionaries into traindata
        self.traindata.update({'ic': train_ic, 'bcl': train_bcl, 
                               'bcr': train_bcr, 'dom': train_dom,
                               'pde': train_pde})


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

class DataBasedNN():
    def __init__(self, data, layers):

        ## gather train data
        self.xtrain = torch.tensor(data['train']['input'][:, 0:1], requires_grad=True).float().to(device)
        self.ttrain = torch.tensor(data['train']['input'][:, 1:2], requires_grad=True).float().to(device)
        self.utrain = torch.tensor(data['train']['output']).float().to(device)

        ## gather test data

        self.xtest = torch.tensor(data['test']['input'][:, 0:1], requires_grad=True).float().to(device)
        self.ttest = torch.tensor(data['test']['input'][:, 1:2], requires_grad=True).float().to(device)
        self.utest = torch.tensor(data['test']['output']).float().to(device)

        ## Assign layers
        self.layers = layers

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(self.dnn.parameters(),
                                           lr=.1,
                                           max_iter=50000,
                                           max_eval=50000,
                                           history_size=50,
                                           tolerance_grad=1e-8,
                                           tolerance_change=1.0 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")
        # self.optimizer = torch.optim.SGD(self.dnn.parameters(),
        #                                  lr=1e-4)

        self.iter = 0
        self.losses = {'train': [],
                       'test': [],
                       'iter': []}

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def loss_func(self):

        self.optimizer.zero_grad()

        uhat_train = self.net_u(self.xtrain, self.ttrain)
        uhat_test = self.net_u(self.xtest, self.ttest)

        loss_train = torch.mean((self.utrain - uhat_train) ** 2)
        loss_test = torch.mean((self.utest - uhat_test) ** 2)


        loss_train.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            self.losses['train'].append(loss_train.item())
            self.losses['test'].append(loss_test.item())
            self.losses['iter'].append(self.iter)
            print(
                'Iter %d, Loss train: %.5e, Loss test: %.5e' % (self.iter, loss_train.item(), loss_test.item())
            )
        return loss_train

    def train(self):
        self.dnn.train()

        # Backward and optimize
        self.optimizer.step(self.loss_func)

        ## Plot the losses
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(self.losses['iter'], self.losses['train'], label='train')
        ax.plot(self.losses['iter'], self.losses['test'], label='test')
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('loss')
        ax.legend()
        plt.savefig('results/losses.png')


    def predict(self, data, xlocs, split, extent):

        ## Obtain the input for the whole domain
        input = data.data['all']['input']
        x = torch.tensor(input[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(input[:, 1:2], requires_grad=True).float().to(device)

        ## Get the evaluation on the domain
        self.dnn.eval()
        uhat = self.net_u(x, t)
        uhat = uhat.detach().cpu().numpy()

        ## Get the ground data
        Ugrnd = data.sol['u']

        ## Extract the x and time data to convert into grid data
        X, T = np.meshgrid(data.sol['x'], data.sol['t'])

        ## Get the grid evaluation of prediction
        Uhat = griddata(input,
                        uhat.flatten(),
                        (X, T), method='cubic')

        ## Evaluate error on the domain
        error = np.abs(Uhat - Ugrnd)

        ################# PLOTTING ###################

        ## Show the location of certain slices ##
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, nrows=1, figsize=(18, 5))
            ## Generate the contour
        im = ax1.imshow(Ugrnd, origin='lower', aspect='auto',
                        extent=np.asarray(extent).flatten(), vmin=Ugrnd.min(),
                        vmax=Ugrnd.max(), interpolation='bilinear', cmap='rainbow')
        clb = fig.colorbar(im, ax=ax1, location='right')
        clb.ax.set_title('T')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('Actual, test/train split and slices', fontsize=12)
            ## Generate and plot the lines
        line = np.linspace(t.detach().numpy().min(), t.detach().numpy().max(), 2).reshape(-1,1)
        nx = data.sol['x'].shape[0]
        nt = data.sol['t'].shape[0]
        colors = plt.cm.brg(np.linspace(0,1,len(xlocs)))
        for i, xloc in enumerate(xlocs):
            ax1.plot(data.sol['x'][int(xloc*nx)]*np.ones((2,1)), line, 
                     linestyle='--',linewidth=1,color=colors[i], clip_on=False)

            ## Show the train and test split on the domain
        t_split = data.sol['t'][int(split*nt)]
        x_test = data.sol['x'][int(0.4*nx)]
        x_train = data.sol['x'][int(0.6*nx)]
        ax1.plot(data.sol['x'], t_split*np.ones((nx,1)), 'g--')
        ax1.arrow(x_test, t_split, 0, 0.5, width=0.02, facecolor='r', alpha=1)
        ax1.arrow(x_train, t_split, 0, -0.5, width=0.02, facecolor='k', alpha=1)
        ax1.annotate('Test', xy=(x_test-0.05, t_split+0.7), fontsize=12, color = 'r')
        ax1.annotate('Train', xy=(x_train-0.05, t_split-0.8), fontsize=12, color = 'k')
            ## Show the output variable along certain slices ##
        for i, xloc in enumerate(xlocs):
            ax2.plot(data.sol['t'], Ugrnd[:,int(xloc*nx)], c=colors[i], linewidth=2, alpha=0.7)
            ax2.plot(data.sol['t'], Uhat[:,int(xloc*nx)], c=colors[i], linestyle='--', linewidth=2)
        ax2.axvline(t_split, color='g', linestyle='--')
        ax2.arrow(t_split, -0.1, 0.5, 0, width=0.02, facecolor='r', alpha=1)
        ax2.arrow(t_split, -0.2, -0.5, 0, width=0.02, facecolor='k', alpha=1)
        ax2.set_xlabel('t')
        ax2.set_ylabel('T')
        lines=ax2.get_lines()
        legend1 = Legend(ax2, lines[:2], ['actual', 'predicted'], loc='upper right')
        ax2.add_artist(legend1)
        legend2 = Legend(ax2, lines[:2*len(xlocs):2], ['x='+str(xloc) for xloc in xlocs], loc='lower left')
        ax2.add_artist(legend2)
        ax2.set_title('solution along slices',fontsize=12)
            ## Show the error in result along slices
        for i, xloc in enumerate(xlocs):
            ax3.plot(data.sol['t'], np.abs(Ugrnd[:,int(xloc*nx)] - Uhat[:,int(xloc*nx)]), c=colors[i], linewidth=2)
        ax3.axvline(t_split, color='g', linestyle='--')
        ax3.arrow(t_split, -0.1, 0.5, 0, width=0.02, facecolor='r', alpha=1)
        ax3.arrow(t_split, -0.2, -0.5, 0, width=0.02, facecolor='k', alpha=1)
        ax3.set_xlabel('t')
        ax3.set_ylabel('T')
        lines=ax3.get_lines()
        legend2 = Legend(ax3, lines[:len(xlocs):1], ['x='+str(xloc) for xloc in xlocs], loc='upper left')
        ax3.add_artist(legend2)
        ax3.set_title('error along slices',fontsize=12)
            ## Save the plot
        plt.savefig('results/slices.png')


        ## Plot the ground truth, predictions and error ##
        fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, nrows=1, figsize=(18, 5))
            ## Ground Truth
        im1 = ax1.imshow(Ugrnd, origin='lower', aspect='auto',
                        extent=np.asarray(extent).flatten(), vmin=Ugrnd.min(),
                        vmax=Ugrnd.max(), interpolation='bilinear', cmap='rainbow')
        ax1.plot(data.sol['x'], t_split*np.ones((nx,1)), 'g--')
        ax1.arrow(x_test, t_split, 0, 0.5, width=0.02, facecolor='r', alpha=1)
        ax1.arrow(x_train, t_split, 0, -0.5, width=0.02, facecolor='k', alpha=1)
        clb1 = fig.colorbar(im1, ax=ax1, location='right')
        clb1.ax.set_title('T')
        ax1.set_title('Actual', fontsize=12)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
            ## Prediction
        im2 = ax2.imshow(Uhat, origin='lower', aspect='auto',
                        extent=np.asarray(extent).flatten(), vmin=Ugrnd.min(),
                        vmax=Ugrnd.max(), interpolation='bilinear', cmap='rainbow')
        ax2.plot(data.sol['x'], t_split*np.ones((nx,1)), 'g--')
        ax2.arrow(x_test, t_split, 0, 0.5, width=0.02, facecolor='r', alpha=1)
        ax2.arrow(x_train, t_split, 0, -0.5, width=0.02, facecolor='k', alpha=1)
        clb2 = fig.colorbar(im2, ax=ax2, location='right')
        clb2.ax.set_title('T')
        ax2.set_title('Prediction', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
            ## Error
        im3 = ax3.imshow(error, origin='lower', aspect='auto',
                        extent=np.asarray(extent).flatten(),
                        interpolation='bilinear', cmap='rainbow')
        ax3.plot(data.sol['x'], t_split*np.ones((nx,1)), 'g--')
        ax3.arrow(x_test, t_split, 0, 0.5, width=0.02, facecolor='r', alpha=1)
        ax3.arrow(x_train, t_split, 0, -0.5, width=0.02, facecolor='k', alpha=1)
        clb3 = fig.colorbar(im3, ax=ax3, location='right')        
        clb3.ax.set_title('T')
        ax3.set_title(r'Error = $|Actual - Prediction|$', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')        
            ## Save the plot
        plt.savefig('results/contours.png')