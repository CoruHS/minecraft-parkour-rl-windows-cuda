package coru.minecraftparkourrl.client.mixin;

import coru.minecraftparkourrl.client.MinecraftParkourRLClient;
import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.Constant;
import org.spongepowered.asm.mixin.injection.ModifyConstant;

@Mixin(MinecraftServer.class)
public class MinecraftServerTickrateMixin {
	@ModifyConstant(method = "runServer", constant = @Constant(longValue = 50L))
	private long minecraftparkourrl$modifyTickWaitTime(long original) {
		return MinecraftParkourRLClient.getConfiguredMillisecondsPerTick();
	}
}
