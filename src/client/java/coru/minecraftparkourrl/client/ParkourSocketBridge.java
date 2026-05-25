package coru.minecraftparkourrl.client;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import coru.minecraftparkourrl.MinecraftParkourRL;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public final class ParkourSocketBridge {
	public static final int PORT = 5005;

	private static final AtomicBoolean started = new AtomicBoolean(false);
	private static final AtomicReference<ActionState> latestAction = new AtomicReference<>(ActionState.inactive());
	private static final ConcurrentLinkedQueue<String> pendingCommands = new ConcurrentLinkedQueue<>();
	private static final Object writerLock = new Object();
	private static PrintWriter telemetryWriter;

	private ParkourSocketBridge() {
	}

	public static void start() {
		if (!started.compareAndSet(false, true)) {
			return;
		}

		Thread socketThread = new Thread(ParkourSocketBridge::runSocketServer, "MinecraftParkourRL-Socket");
		socketThread.setDaemon(true);
		socketThread.start();
	}

	public static ActionState getLatestAction() {
		return latestAction.get();
	}

	public static String pollCommand() {
		return pendingCommands.poll();
	}

	public static void sendTelemetry(JsonObject telemetry) {
		synchronized (writerLock) {
			if (telemetryWriter == null) {
				return;
			}

			telemetryWriter.println(telemetry);
			if (telemetryWriter.checkError()) {
				telemetryWriter.close();
				telemetryWriter = null;
				latestAction.set(ActionState.inactive());
			}
		}
	}

	private static void runSocketServer() {
		try (ServerSocket serverSocket = new ServerSocket()) {
			serverSocket.setReuseAddress(true);
			serverSocket.bind(new InetSocketAddress(InetAddress.getLoopbackAddress(), PORT));
			MinecraftParkourRL.LOGGER.info("Minecraft Parkour RL socket listening on 127.0.0.1:{}", PORT);

			while (true) {
				Socket socket = serverSocket.accept();
				handleClient(socket);
			}
		} catch (IOException exception) {
			MinecraftParkourRL.LOGGER.error("Minecraft Parkour RL socket server stopped", exception);
		}
	}

	private static void handleClient(Socket socket) {
		MinecraftParkourRL.LOGGER.info("Minecraft Parkour RL Python client connected from {}", socket.getRemoteSocketAddress());
		latestAction.set(ActionState.released());

		try (
				Socket clientSocket = socket;
				BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(), StandardCharsets.UTF_8));
				PrintWriter writer = new PrintWriter(clientSocket.getOutputStream(), true, StandardCharsets.UTF_8)
		) {
			synchronized (writerLock) {
				telemetryWriter = writer;
			}

			String line;
			while ((line = reader.readLine()) != null) {
				handleMessage(line);
			}
		} catch (IOException exception) {
			MinecraftParkourRL.LOGGER.warn("Minecraft Parkour RL Python client disconnected: {}", exception.getMessage());
		} finally {
			synchronized (writerLock) {
				telemetryWriter = null;
			}
			latestAction.set(ActionState.inactive());
		}
	}

	private static void handleMessage(String line) {
		if (line.isBlank()) {
			return;
		}

		try {
			JsonObject message = JsonParser.parseString(line).getAsJsonObject();
			if (message.has("command")) {
				pendingCommands.add(message.get("command").getAsString());
			}
			latestAction.set(ActionState.fromJson(message));
		} catch (RuntimeException exception) {
			MinecraftParkourRL.LOGGER.warn("Invalid parkour action JSON: {}", line);
		}
	}

	private static boolean getBoolean(JsonObject object, String key) {
		return object.has(key) && object.get(key).getAsBoolean();
	}

	private static double getDouble(JsonObject object, String key) {
		return object.has(key) ? object.get(key).getAsDouble() : 0.0D;
	}

	public record ActionState(
			boolean active,
			boolean forward,
			boolean back,
			boolean left,
			boolean right,
			boolean jump,
			boolean sprint,
			double yawDelta,
			double pitchDelta
	) {
		public static ActionState inactive() {
			return new ActionState(false, false, false, false, false, false, false, 0.0D, 0.0D);
		}

		public static ActionState released() {
			return new ActionState(true, false, false, false, false, false, false, 0.0D, 0.0D);
		}

		public static ActionState fromJson(JsonObject object) {
			return new ActionState(
					true,
					getBoolean(object, "forward"),
					getBoolean(object, "back"),
					getBoolean(object, "left"),
					getBoolean(object, "right"),
					getBoolean(object, "jump"),
					getBoolean(object, "sprint"),
					getDouble(object, "yaw_delta"),
					getDouble(object, "pitch_delta")
			);
		}
	}
}
