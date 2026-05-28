package coru.minecraftparkourrl.client;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.systems.RenderSystem;
import coru.minecraftparkourrl.MinecraftParkourRL;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.gl.Framebuffer;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.input.Input;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.render.RenderTickCounter;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.GameMode;
import coru.minecraftparkourrl.client.mixin.MinecraftClientAccessor;
import coru.minecraftparkourrl.client.mixin.RenderTickCounterAccessor;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL30;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

public class MinecraftParkourRLClient implements ClientModInitializer {
	private static final float TICKRATE = 20.0F;
	private static final String WORLD_FOLDER = "ParkourRL";
	private static final int LAN_PORT = 25565;
	private static final String LAN_PORT_FILE = "parkour_lan_port.txt";
	// frame capture size: 84x84 is standard for RL atari-style input
	private static final int FRAME_W = 84;
	private static final int FRAME_H = 84;

	private static float lastTickrate = -1.0F;
	private static boolean triedOpenWorld;
	private static boolean triedOpenLan;
	private static boolean warnedCapture;

	// Reusable frame-capture buffers. Allocating a fresh full-window direct
	// ByteBuffer every tick (40/s) generated huge off-heap churn that forced
	// escalating GC and throttled the render/tick loop over time. We allocate
	// the readback buffer once and only grow it if the window gets bigger.
	private static ByteBuffer captureBuffer;
	private static int captureBufferCapacity = -1;
	private static final byte[] downsampledRgb = new byte[FRAME_W * FRAME_H * 3];

	@Override
	public void onInitializeClient() {
		ClientTickEvents.START_CLIENT_TICK.register(MinecraftParkourRLClient::applyTickrate);
		ClientTickEvents.START_CLIENT_TICK.register(MinecraftParkourRLClient::tryOpenWorld);
		ClientTickEvents.START_CLIENT_TICK.register(MinecraftParkourRLClient::tryOpenLan);
		ClientTickEvents.START_CLIENT_TICK.register(MinecraftParkourRLClient::applySocketAction);
		ClientTickEvents.END_CLIENT_TICK.register(MinecraftParkourRLClient::outputTelemetry);
		ParkourSocketBridge.start();
		MinecraftParkourRL.LOGGER.info("ParkourRL client ready");
	}

	public static float getConfiguredTickrate() {
		return TICKRATE;
	}

	public static long getConfiguredMillisecondsPerTick() {
		return Math.max(1L, Math.round(1000.0F / getConfiguredTickrate()));
	}

	private static void applyTickrate(MinecraftClient client) {
		float tickrate = getConfiguredTickrate();
		if (tickrate == lastTickrate) return;

		RenderTickCounter counter = ((MinecraftClientAccessor) client).minecraftparkourrl$getRenderTickCounter();
		((RenderTickCounterAccessor) counter).minecraftparkourrl$setTickTime(1000.0F / tickrate);
		lastTickrate = tickrate;
		MinecraftParkourRL.LOGGER.info("Tickrate set to {} TPS", tickrate);
	}

	private static void tryOpenWorld(MinecraftClient client) {
		if (triedOpenWorld || client.world != null || client.currentScreen == null) return;

		Path levelDat = client.runDirectory.toPath()
				.resolve("saves").resolve(WORLD_FOLDER).resolve("level.dat");

		if (!Files.isRegularFile(levelDat)) {
			triedOpenWorld = true;
			MinecraftParkourRL.LOGGER.warn("World '{}' not found - create it first", WORLD_FOLDER);
			return;
		}

		triedOpenWorld = true;
		MinecraftParkourRL.LOGGER.info("Opening world '{}'", WORLD_FOLDER);
		client.createIntegratedServerLoader().start(client.currentScreen, WORLD_FOLDER);
	}

	private static void tryOpenLan(MinecraftClient client) {
		if (client.world == null || client.player == null) {
			triedOpenLan = false;
			return;
		}
		if (triedOpenLan || !client.isIntegratedServerRunning() || client.getServer() == null) return;

		// already open from a previous session
		if (client.getServer().getServerPort() > 0) {
			triedOpenLan = true;
			writeLanPort(client, client.getServer().getServerPort());
			return;
		}

		triedOpenLan = true;
		try {
			boolean ok = client.getServer().openToLan(GameMode.SPECTATOR, true, LAN_PORT);
			if (ok) {
				int port = client.getServer().getServerPort();
				writeLanPort(client, port);
				MinecraftParkourRL.LOGGER.info("LAN open on port {}", port);
			} else {
				MinecraftParkourRL.LOGGER.warn("LAN failed on port {}", LAN_PORT);
			}
		} catch (RuntimeException e) {
			MinecraftParkourRL.LOGGER.warn("LAN failed", e);
		}
	}

	// Python reads this file to know what port to connect to.
	private static void writeLanPort(MinecraftClient client, int port) {
		Path file = client.runDirectory.toPath().resolve(LAN_PORT_FILE);
		try {
			Files.writeString(file, Integer.toString(port));
		} catch (IOException e) {
			MinecraftParkourRL.LOGGER.warn("Couldn't write port file", e);
		}
	}

	private static void applySocketAction(MinecraftClient client) {
		ClientPlayerEntity player = client.player;
		if (player == null || client.world == null) return;

		// drain any queued commands (tp, gamemode, etc)
		String cmd = ParkourSocketBridge.pollCommand();
		while (cmd != null) {
			if (client.getNetworkHandler() != null) {
				String clean = cmd.startsWith("/") ? cmd.substring(1) : cmd;
				client.getNetworkHandler().sendChatCommand(clean);
			}
			cmd = ParkourSocketBridge.pollCommand();
		}

		ParkourSocketBridge.ActionState action = ParkourSocketBridge.getLatestAction();
		if (!action.active()) return;

		// set both the keybind state and the input state
		// keybind state drives the HUD display, input state drives actual movement
		client.options.forwardKey.setPressed(action.forward());
		client.options.backKey.setPressed(action.back());
		client.options.leftKey.setPressed(action.left());
		client.options.rightKey.setPressed(action.right());
		client.options.jumpKey.setPressed(action.jump());
		client.options.sprintKey.setPressed(action.sprint());

		player.input.pressingForward = action.forward();
		player.input.pressingBack = action.back();
		player.input.pressingLeft = action.left();
		player.input.pressingRight = action.right();
		player.input.jumping = action.jump();
		player.setSprinting(action.sprint() && action.forward());

		// camera control: pitch clamped so we cannot look beyond straight up/down
		if (action.yawDelta() != 0.0D || action.pitchDelta() != 0.0D) {
			player.setYaw(MathHelper.wrapDegrees(player.getYaw() + (float) action.yawDelta()));
			player.setPitch(MathHelper.clamp(player.getPitch() + (float) action.pitchDelta(), -90.0F, 90.0F));
		}
	}

	private static void outputTelemetry(MinecraftClient client) {
		ClientPlayerEntity player = client.player;
		if (player == null || client.world == null) return;

		JsonObject telemetry = new JsonObject();
		telemetry.addProperty("world_time", client.world.getTime());
		telemetry.addProperty("sprinting", player.isSprinting());
		telemetry.addProperty("tickrate", getConfiguredTickrate());
		telemetry.addProperty("configured_tickrate", getConfiguredTickrate());
		telemetry.addProperty("lan_open", isLanOpen(client));
		telemetry.addProperty("lan_port", getLanPort(client));

		// position
		JsonObject pos = new JsonObject();
		pos.addProperty("x", player.getX());
		pos.addProperty("y", player.getY());
		pos.addProperty("z", player.getZ());
		telemetry.add("position", pos);

		// rotation
		JsonObject rot = new JsonObject();
		rot.addProperty("yaw", MathHelper.wrapDegrees(player.getYaw()));
		rot.addProperty("pitch", player.getPitch());
		telemetry.add("rotation", rot);

		// movement keys
		JsonObject move = new JsonObject();
		Input input = player.input;
		move.addProperty("forward", input.pressingForward);
		move.addProperty("back", input.pressingBack);
		move.addProperty("left", input.pressingLeft);
		move.addProperty("right", input.pressingRight);
		telemetry.add("movement", move);

		// the 14-element vector the neural network actually eats
		telemetry.add("mlp_state", buildMlpState(client, player));

		// frame capture for CNN; skip if not on render thread
		JsonObject frame = captureFrame(client);
		if (frame != null) telemetry.add("frame", frame);

		ParkourSocketBridge.sendTelemetry(telemetry);
	}

	private static boolean isLanOpen(MinecraftClient client) {
		return getLanPort(client) > 0;
	}

	private static int getLanPort(MinecraftClient client) {
		if (!client.isIntegratedServerRunning() || client.getServer() == null) {
			return -1;
		}

		return client.getServer().getServerPort();
	}

	private static JsonArray buildMlpState(MinecraftClient client, ClientPlayerEntity player) {
		JsonArray state = new JsonArray();
		Vec3d vel = player.getVelocity();
		Input input = player.input;
		double yawRad = Math.toRadians(MathHelper.wrapDegrees(player.getYaw()));

		state.add(Math.sin(yawRad));
		state.add(Math.cos(yawRad));
		state.add(player.getPitch() / 90.0F);
		state.add(vel.x);
		state.add(vel.y);
		state.add(vel.z);
		state.add(player.isOnGround() ? 1.0 : 0.0);
		state.add(player.isSprinting() ? 1.0 : 0.0);
		state.add(input.pressingForward ? 1.0 : 0.0);
		state.add(input.pressingBack ? 1.0 : 0.0);
		state.add(input.pressingLeft ? 1.0 : 0.0);
		state.add(input.pressingRight ? 1.0 : 0.0);
		state.add(input.jumping ? 1.0 : 0.0);
		state.add(client.options.sprintKey.isPressed() ? 1.0 : 0.0);

		return state;
	}

	private static JsonObject captureFrame(MinecraftClient client) {
		if (!RenderSystem.isOnRenderThread()) {
			if (!warnedCapture) {
				MinecraftParkourRL.LOGGER.warn("Frame capture skipped - not on render thread");
				warnedCapture = true;
			}
			return null;
		}

		Framebuffer fb = client.getFramebuffer();
		int srcW = fb.viewportWidth;
		int srcH = fb.viewportHeight;
		if (srcW <= 0 || srcH <= 0) return null;

		// save GL state so we do not break Minecraft rendering
		int prevFbo = GL11.glGetInteger(GL30.GL_READ_FRAMEBUFFER_BINDING);
		int prevBuf = GL11.glGetInteger(GL11.GL_READ_BUFFER);
		int prevAlign = GL11.glGetInteger(GL11.GL_PACK_ALIGNMENT);

		int required = srcW * srcH * 4;
		if (captureBuffer == null || captureBufferCapacity < required) {
			captureBuffer = BufferUtils.createByteBuffer(required);
			captureBufferCapacity = required;
		}
		ByteBuffer rgba = captureBuffer;
		rgba.clear();

		try {
			GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, fb.fbo);
			GL11.glReadBuffer(GL30.GL_COLOR_ATTACHMENT0);
			GL11.glPixelStorei(GL11.GL_PACK_ALIGNMENT, 1);
			GL11.glReadPixels(0, 0, srcW, srcH, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, rgba);
		} catch (RuntimeException e) {
			if (!warnedCapture) {
				MinecraftParkourRL.LOGGER.warn("Frame capture failed", e);
				warnedCapture = true;
			}
			return null;
		} finally {
			GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, prevFbo);
			GL11.glReadBuffer(prevBuf);
			GL11.glPixelStorei(GL11.GL_PACK_ALIGNMENT, prevAlign);
		}

		// downsample to 84x84 RGB; nearest neighbor is fine for RL
		byte[] rgb = downsampledRgb;
		for (int y = 0; y < FRAME_H; y++) {
			int srcY = srcH - 1 - (y * srcH / FRAME_H); // flip vertically; GL is bottom-up
			for (int x = 0; x < FRAME_W; x++) {
				int srcX = x * srcW / FRAME_W;
				int srcOff = (srcY * srcW + srcX) * 4;
				int dstOff = (y * FRAME_W + x) * 3;
				rgb[dstOff] = rgba.get(srcOff);
				rgb[dstOff + 1] = rgba.get(srcOff + 1);
				rgb[dstOff + 2] = rgba.get(srcOff + 2);
			}
		}

		JsonObject frame = new JsonObject();
		frame.addProperty("width", FRAME_W);
		frame.addProperty("height", FRAME_H);
		frame.addProperty("channels", 3);
		frame.addProperty("format", "rgb_u8_base64");
		frame.addProperty("data", Base64.getEncoder().encodeToString(rgb));
		return frame;
	}
}
