
do  
    local Visualize = torch.class('Visualize')

    function Visualize:__init()
        
    end

    function Visualize:plotLossFigure(losses,losses_iter,val_losses,val_losses_iter,out_file_loss_plot) 
        local ff=gnuplot.pngfigure(out_file_loss_plot)
        -- print (out_file_loss_plot)
        local losses_tensor = torch.Tensor{losses_iter,losses};
        if #val_losses>0 then
            local val_losses_tensor=torch.Tensor{val_losses_iter,val_losses}
            gnuplot.plot({'Train Loss',losses_tensor[1],losses_tensor[2]},{'Val Loss',val_losses_tensor[1],val_losses_tensor[2]});
            gnuplot.grid(true)
        else
            gnuplot.plot({'Train Loss ',losses_tensor[1],losses_tensor[2]});

        end
        gnuplot.title('Losses'..losses_iter[#losses_iter])
        gnuplot.xlabel('Iterations');
        gnuplot.ylabel('Loss');
        gnuplot.plotflush();
        gnuplot.closeall();
        -- gnuplot.pngfigure(out_file_loss_plot);
    end

    function Visualize:plotHist(vals,n_bins,out_file) 
        gnuplot.pngfigure(out_file)
        local str_shape='';
        for idx_size_curr=1,#vals:size() do
            local size_curr = vals:size()[idx_size_curr];

            str_shape=str_shape..' '..size_curr;
        end
        gnuplot.hist(vals:view(vals:nElement()),n_bins)
        gnuplot.title('Parameters'..str_shape)
        gnuplot.xlabel('Values');
        gnuplot.ylabel('Frequency');
        gnuplot.plotflush();
        gnuplot.close();
        -- gnuplot.pngfigure(out_file_loss_plot);
    end


end    

return Visualize;