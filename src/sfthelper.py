import torch
import logging
import ipdb
import os
from tqdm import tqdm
import numpy as np

from src.utils import save_checkpoint

logger = logging.getLogger(__name__)

class TrainingHelper:
    """Helper class for training functions"""
    
    @staticmethod
    def train_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch, num_epochs):
        """Run one epoch of training"""
        model.train()
        epoch_loss = 0
        epoch_metrics = {
            'distillation_loss': 0,
            'ce_loss': 0,
            'switch_loss': 0,
        }
        num_steps = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            output = model(batch["input_ids"])
            loss_dict = model.compute_loss(
                batch,
                output
            )
            
            loss = loss_dict['total_loss']
            accelerator.backward(loss)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Accumulate losses and metrics
            epoch_loss += loss.detach().float()
            epoch_metrics['distillation_loss'] += loss_dict['distillation_loss'].item()
            epoch_metrics['ce_loss'] += loss_dict['ce_loss'].item()
            epoch_metrics['switch_loss'] += loss_dict['switch_loss'].item()
            num_steps += 1
            
            if accelerator.is_local_main_process:
                metrics = {
                    "train/current_epoch": epoch + 1,  # Add current epoch number
                    "train/total_epochs": num_epochs,  # Add total epochs
                    "train/step": num_steps,  # Add current step within epoch
                    "train/step_loss": loss.item(),
                    "train/step_distillation_loss": loss_dict['distillation_loss'].item(),
                    "train/step_ce_loss": loss_dict['ce_loss'].item(),
                    "train/step_switch_loss": loss_dict['switch_loss'].item(),
                    "train/step_learning_rate": scheduler.get_last_lr()[0],
                }
                accelerator.log(metrics)
                
                progress_bar.set_postfix({
                    "epoch": f"{epoch + 1}/{num_epochs}",  # Add epoch to progress bar
                    "step": num_steps,  # Add step to progress bar
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        # Calculate epoch averages
        avg_metrics = {k: v / num_steps for k, v in epoch_metrics.items()} if num_steps > 0 else epoch_metrics
        avg_loss = epoch_loss / num_steps if num_steps > 0 else float('inf')
        
        return avg_loss, avg_metrics

    @staticmethod
    def validate(model, dataloader, accelerator):
        """Run validation with enhanced metrics tracking"""
        model.eval()
        total_loss = 0
        val_metrics = {
            'distillation_loss': 0,
            'ce_loss': 0,
            'switch_loss': 0,
        }
        num_steps = 0
        
        progress_bar = tqdm(
            dataloader,
            desc="Validating",
            disable=not accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                output = model(batch["input_ids"])
                loss_dict = model.compute_loss(
                    batch,
                    output
                )
                
                # Accumulate all metrics
                total_loss += loss_dict['total_loss'].item()
                val_metrics['distillation_loss'] += loss_dict['distillation_loss'].item()
                val_metrics['ce_loss'] += loss_dict['ce_loss'].item()
                val_metrics['switch_loss'] += loss_dict['switch_loss'].item()
                num_steps += 1
                
                if accelerator.is_local_main_process:
                    # Log current batch metrics
                    metrics = {
                        "val/current_loss": loss_dict['total_loss'].item(),
                        "val/current_distillation_loss": loss_dict['distillation_loss'].item(),
                        "val/current_ce_loss": loss_dict['ce_loss'].item(),
                        "val/current_switch_loss": loss_dict['switch_loss'].item(),
                    }
                    accelerator.log(metrics)
                    
                    progress_bar.set_postfix({
                        "val_loss": f"{total_loss/num_steps:.4f}",
                    })
        
        # Calculate validation averages
        avg_metrics = {k: v / num_steps for k, v in val_metrics.items()} if num_steps > 0 else val_metrics
        avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
        
        return avg_loss, avg_metrics

    @staticmethod
    def train(model, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, args, output_dirs):
        """Main training loop with enhanced metrics tracking"""
        logger.info("Starting training")
        
        best_model = accelerator.unwrap_model(model).state_dict()
        best_loss = float('inf')
        
        # Save initial checkpoint before training
        if accelerator.is_local_main_process:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=-1,
                save_dir=os.path.join(output_dirs['checkpoints'], "checkpoint-0"),
                metrics={},
                accelerator=accelerator
            )
        
        for epoch in range(args.num_epochs):
            # Training phase
            train_loss, train_metrics = TrainingHelper.train_epoch(
                model=model, 
                dataloader=train_dataloader, 
                optimizer=optimizer, 
                scheduler=scheduler,
                accelerator=accelerator, 
                epoch=epoch, 
                num_epochs=args.num_epochs
            )
            
            # Validation phase
            val_loss, val_metrics = TrainingHelper.validate(
                model=model,
                dataloader=val_dataloader,
                accelerator=accelerator
            )
            
            # Save periodic checkpoints
            if ((epoch + 1) % args.save_epochs == 0 or epoch < 10) and accelerator.is_local_main_process:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    epoch=epoch,
                    save_dir=os.path.join(output_dirs['checkpoints'], f"checkpoint-{epoch+1}"),
                    metrics={"val_loss": val_loss, **val_metrics},
                    accelerator=accelerator
                )
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = accelerator.unwrap_model(model).state_dict()
                
                if accelerator.is_local_main_process:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        save_dir=os.path.join(output_dirs['checkpoints'], "best"),
                        metrics={"val_loss": val_loss, **val_metrics},
                        accelerator=accelerator
                    )
            
            if accelerator.is_local_main_process:
                # Log comprehensive epoch metrics
                accelerator.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/epoch_distillation_loss": train_metrics['distillation_loss'],
                    "train/epoch_ce_loss": train_metrics['ce_loss'],
                    "train/epoch_switch_loss": train_metrics['switch_loss'],
                    "val/epoch_loss": val_loss,
                    "val/epoch_distillation_loss": val_metrics['distillation_loss'],
                    "val/epoch_ce_loss": val_metrics['ce_loss'],
                    "val/epoch_switch_loss": val_metrics['switch_loss'],
                })
                
                logger.info(
                    f"Epoch {epoch + 1}/{args.num_epochs} - "
                    f"Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                )
        
        model.load_state_dict(best_model)
        return model